"""Kafka worker độc lập xử lý image optimization event theo từng partition/message.

Message hợp lệ chỉ được commit sau khi job đã được claim và persistence hoàn tất. Event
không phục hồi được đi DLQ; exception hạ tầng tạm thời giữ nguyên offset để broker giao lại.
"""

import asyncio
import json
import logging
from collections.abc import Mapping
from uuid import uuid4

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import ConsumerRecord, OffsetAndMetadata, TopicPartition
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.bootstrap.lifespan import validate_runtime_settings
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.modules.image_optimization.application.events import ImageOptimizationRequestedEvent
from app.modules.image_optimization.application.processor import ImageOptimizationJobProcessor
from app.modules.image_optimization.infrastructure.clients import HttpMediaAssetClient
from app.modules.image_optimization.infrastructure.persistence.sqlalchemy_repository import (
    SqlAlchemyImageOptimizationJobRepository,
)
from app.modules.image_optimization.infrastructure.providers.openai_image import OpenAILifestyleImageProvider
from app.modules.image_optimization.infrastructure.providers.white_background import WhiteBackgroundProvider
from app.modules.image_optimization.infrastructure.security import FernetBackgroundDescriptionCipher

logger = logging.getLogger(__name__)


# Parse event versioned; payload lỗi được xem là không phục hồi được và phải vào DLQ.
def _parse_event(value: bytes) -> ImageOptimizationRequestedEvent:
    """Không log payload raw vì nó chứa seller/product metadata."""

    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Kafka event must be an object")
    return ImageOptimizationRequestedEvent.from_payload(payload)


# Xử lý một event trong session riêng để lỗi của message này không rollback message khác.
async def _process_event(
    event: ImageOptimizationRequestedEvent,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    processor_factory: object,
) -> None:
    """Commit state sau processor; atomic lease ngăn redelivery gọi provider lần hai."""

    if not callable(processor_factory):
        raise TypeError("Processor factory is not callable")
    async with session_factory() as session:
        repository = SqlAlchemyImageOptimizationJobRepository(session)
        processor = processor_factory(repository)
        if not isinstance(processor, ImageOptimizationJobProcessor):
            raise TypeError("Processor factory returned an invalid processor")
        await processor.execute(event.job_id)
        await session.commit()


# Gửi event không parse được vào DLQ, giữ nguyên bytes để đội vận hành có thể điều tra schema.
async def _send_dlq(producer: AIOKafkaProducer, topic: str, message: ConsumerRecord[bytes, bytes]) -> None:
    """Dùng key gốc nếu có; không ghi payload hoặc exception detail vào log."""

    await producer.send_and_wait(topic, value=message.value, key=message.key)


# Xử lý tuần tự trong một partition để chỉ commit offset liên tục đã hoàn tất.
async def _process_partition(
    partition: TopicPartition,
    messages: list[ConsumerRecord[bytes, bytes]],
    *,
    consumer: AIOKafkaConsumer,
    producer: AIOKafkaProducer,
    dlq_topic: str,
    session_factory: async_sessionmaker[AsyncSession],
    processor_factory: object,
) -> None:
    """Dừng partition ở lỗi tạm thời để message sau không bị commit vượt qua message lỗi."""

    for message in messages:
        try:
            event = _parse_event(message.value)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            await _send_dlq(producer, dlq_topic, message)
        else:
            try:
                await _process_event(event, session_factory=session_factory, processor_factory=processor_factory)
            except Exception:
                logger.exception("Image optimization message failed before commit")
                break
        await consumer.commit({partition: OffsetAndMetadata(message.offset + 1, "")})


# Khởi tạo shared clients/providers một lần rồi consume các partition song song có giới hạn.
async def run_worker() -> None:
    """Fail-fast khi production dependency thiếu và đóng toàn bộ pool khi shutdown."""

    configure_logging()
    settings = get_settings()
    validate_runtime_settings(settings)
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required for image optimization worker")
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    consumer = AIOKafkaConsumer(
        settings.kafka_image_optimization_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        enable_auto_commit=False,
    )
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(30.0, connect=5.0)) as http_client:
        media_client = HttpMediaAssetClient(settings, http_client)
        white_provider = WhiteBackgroundProvider(
            max_dimension=settings.ai_image_max_dimension,
            webp_quality=settings.ai_image_webp_quality,
        )
        await white_provider.warm_up()
        lifestyle_provider = OpenAILifestyleImageProvider(settings, http_client)
        cipher_secret = settings.ai_image_background_encryption_key
        cipher = FernetBackgroundDescriptionCipher(cipher_secret.get_secret_value()) if cipher_secret else None
        worker_id = f"image-worker-{uuid4()}"

        # Factory nhận repository theo message, còn provider/client pool được dùng chung suốt process.
        def processor_factory(repository: SqlAlchemyImageOptimizationJobRepository) -> ImageOptimizationJobProcessor:
            """Tạo processor transaction-scoped nhưng không tạo lại network client/provider."""

            return ImageOptimizationJobProcessor(
                repository=repository,
                media_client=media_client,
                white_provider=white_provider,
                lifestyle_provider=lifestyle_provider,
                max_retry_attempts=settings.ai_image_max_retry_attempts,
                background_cipher=cipher,
                worker_id=worker_id,
                lease_seconds=settings.ai_image_job_lease_seconds,
                retention_days=settings.ai_image_review_retention_days,
            )

        await producer.start()
        await consumer.start()
        try:
            while True:
                records = await consumer.getmany(
                    timeout_ms=settings.ai_image_kafka_poll_timeout_ms,
                    max_records=settings.ai_image_worker_concurrency,
                )
                await asyncio.gather(
                    *(
                        _process_partition(
                            partition,
                            list(messages),
                            consumer=consumer,
                            producer=producer,
                            dlq_topic=settings.kafka_image_optimization_dlq_topic,
                            session_factory=session_factory,
                            processor_factory=processor_factory,
                        )
                        for partition, messages in records.items()
                    )
                )
        finally:
            await consumer.stop()
            await producer.stop()
    await engine.dispose()


# Cung cấp CLI module ổn định cho npm run worker.
def main() -> None:
    """Chạy worker và xử lý Ctrl+C như shutdown bình thường."""

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Image optimization worker stopped")


if __name__ == "__main__":
    main()
