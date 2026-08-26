"""Kafka entrypoint cho image optimization worker.

Process nay chay doc lap voi FastAPI. Moi message chi chua job metadata; worker mo session
PostgreSQL rieng, xu ly idempotent va commit sau khi output da duoc Media Service luu.
"""

import asyncio
import json
import logging
from contextlib import suppress
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.modules.image_optimization.application.processor import ImageOptimizationJobProcessor
from app.modules.image_optimization.infrastructure.clients import HttpMediaAssetClient
from app.modules.image_optimization.infrastructure.persistence.outbox_relay import SqlAlchemyOutboxRelay
from app.modules.image_optimization.infrastructure.persistence.sqlalchemy_repository import (
    SqlAlchemyImageOptimizationJobRepository,
)
from app.modules.image_optimization.infrastructure.providers.openai_image import OpenAILifestyleImageProvider
from app.modules.image_optimization.infrastructure.providers.white_background import WhiteBackgroundProvider
from app.modules.image_optimization.infrastructure.security import FernetBackgroundDescriptionCipher

logger = logging.getLogger(__name__)


# Parse và xử lý một event bằng dependency dùng chung, chỉ commit DB sau khi output hoàn tất.
async def _process_message(
    message_value: bytes,
    session_factory: async_sessionmaker[AsyncSession],
    media_client: HttpMediaAssetClient,
    white_provider: WhiteBackgroundProvider,
    lifestyle_provider: OpenAILifestyleImageProvider | None,
    background_cipher: FernetBackgroundDescriptionCipher | None,
) -> None:
    """Parse event va xu ly trong transaction rieng; payload loi se bi bo qua an toan."""

    settings = get_settings()
    try:
        event = json.loads(message_value.decode("utf-8"))
        job_id = UUID(str(event["jobId"]))
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        logger.error("Invalid image optimization event: %s", type(error).__name__)
        return
    if not settings.database_url:
        logger.error("AI worker requires DATABASE_URL; message was not processed")
        return
    async with session_factory() as session:
        repository = SqlAlchemyImageOptimizationJobRepository(session)
        processor = ImageOptimizationJobProcessor(
            repository=repository,
            media_client=media_client,
            white_provider=white_provider,
            lifestyle_provider=lifestyle_provider,
            max_retry_attempts=settings.ai_image_max_retry_attempts,
            background_cipher=background_cipher,
        )
        await processor.execute(job_id)
        await session.commit()


# Khởi động Kafka/DB/client pool một lần và xử lý batch message song song nhưng commit an toàn.
async def run_worker() -> None:
    """Consume Kafka khi broker va database da duoc cau hinh; local thieu infra thi thoat ro rang."""

    settings = get_settings()
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        from aiokafka.structs import OffsetAndMetadata, TopicPartition
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    except ImportError:
        logger.error("aiokafka is not installed; install requirements before running the worker")
        return
    if not settings.database_url:
        logger.error("DATABASE_URL is required for the image optimization worker")
        return

    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    consumer = AIOKafkaConsumer(
        settings.kafka_image_optimization_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        enable_auto_commit=False,
        value_deserializer=lambda value: value,
    )
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as http_client:
        media_client = HttpMediaAssetClient(settings, http_client)
        white_provider = WhiteBackgroundProvider(
            max_dimension=settings.ai_image_max_dimension,
            webp_quality=settings.ai_image_webp_quality,
        )
        # Tải model trước khi nhận message để request đầu tiên không bị cộng thêm thời gian warm-up.
        await white_provider.warm_up()
        lifestyle_provider = OpenAILifestyleImageProvider(settings) if settings.openai_api_key else None
        cipher_secret = (
            settings.ai_image_background_encryption_key.get_secret_value()
            if settings.ai_image_background_encryption_key
            else None
        )
        background_cipher = FernetBackgroundDescriptionCipher(cipher_secret) if cipher_secret else None
        await producer.start()
        await consumer.start()
        relay = SqlAlchemyOutboxRelay(session_factory, producer, settings.kafka_image_optimization_topic)

        async def relay_loop() -> None:
            """Relay event sau commit của API, tránh mất message nếu Kafka tạm thời lỗi."""

            while True:
                # Relay sẽ thử lại ở vòng sau; event vẫn còn published_at = NULL nên không mất yêu cầu.
                with suppress(Exception):
                    await relay.relay_once()
                await asyncio.sleep(settings.ai_image_outbox_poll_interval_ms / 1_000)

        relay_task = asyncio.create_task(relay_loop())
        try:
            while True:
                # Lấy một batch nhỏ để xử lý song song; chỉ commit sau khi toàn bộ batch hoàn tất để không mất message.
                records = await consumer.getmany(
                    timeout_ms=settings.ai_image_kafka_poll_timeout_ms,
                    max_records=settings.ai_image_worker_concurrency,
                )
                messages = [message for partition in records.values() for message in partition]
                if not messages:
                    continue
                await asyncio.gather(
                    *(
                        _process_message(
                            message.value,
                            session_factory,
                            media_client,
                            white_provider,
                            lifestyle_provider,
                            background_cipher,
                        )
                        for message in messages
                    )
                )
                offsets = {
                    TopicPartition(message.topic, message.partition): OffsetAndMetadata(message.offset + 1, "")
                    for message in messages
                }
                await consumer.commit(offsets)
        finally:
            relay_task.cancel()
            with suppress(asyncio.CancelledError):
                await relay_task
            await producer.stop()
            await consumer.stop()
    await engine.dispose()


def main() -> None:
    """Chay worker bang lenh npm run worker hoac python -m app.workers.image_optimization_worker."""

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
