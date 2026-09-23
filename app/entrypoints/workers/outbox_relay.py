"""Entrypoint relay outbox PostgreSQL sang Kafka, chạy độc lập với API và worker."""

import asyncio
import logging
import os

from aiokafka import AIOKafkaProducer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.bootstrap.lifespan import validate_runtime_settings
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.worker_health import clear_worker_health, mark_worker_healthy
from app.modules.image_optimization.infrastructure.persistence.outbox_relay import SqlAlchemyOutboxRelay

logger = logging.getLogger(__name__)


# Mở database/Kafka một lần và relay liên tục theo policy retry bền vững trong outbox table.
async def run_outbox_relay() -> None:
    """Không nuốt lỗi ngoài Kafka; supervisor sẽ restart process khi dependency nền tảng hỏng."""

    health_file = os.getenv("WORKER_HEALTH_FILE", "/tmp/bin-ecommerce-ai-outbox-relay.health")
    # Xóa marker cũ trước khi mở producer để startup probe luôn chờ kết nối Kafka mới.
    clear_worker_health(health_file)
    configure_logging()
    settings = get_settings()
    validate_runtime_settings(settings)
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required for outbox relay")
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    await producer.start()
    relay = SqlAlchemyOutboxRelay(
        session_factory,
        producer,
        settings.kafka_image_optimization_topic,
        dlq_topic=settings.kafka_image_optimization_dlq_topic,
        max_attempts=settings.ai_image_outbox_max_attempts,
    )
    try:
        mark_worker_healthy(health_file)
        while True:
            published = await relay.relay_once()
            # relay_once chỉ trả về sau khi database/Kafka đã được kiểm tra thành công.
            mark_worker_healthy(health_file)
            if published == 0:
                await asyncio.sleep(settings.ai_image_outbox_poll_interval_ms / 1_000)
    finally:
        clear_worker_health(health_file)
        await producer.stop()
        await engine.dispose()


# Chạy relay bằng module entrypoint để npm/Makefile không phụ thuộc đường dẫn implementation.
def main() -> None:
    """Khởi chạy event loop và cho phép Ctrl+C dừng sạch."""

    try:
        asyncio.run(run_outbox_relay())
    except KeyboardInterrupt:
        logger.info("Outbox relay stopped")


if __name__ == "__main__":
    main()
