"""Relay outbox PostgreSQL sang Kafka sau khi transaction tao job da commit."""

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.image_optimization.infrastructure.persistence.models import ImageOptimizationOutboxRecord


class MessageProducer(Protocol):
    """Port toi Kafka producer, de relay co the test bang fake."""

    async def send_and_wait(self, topic: str, value: bytes, key: bytes | None = None) -> object:
        """Gui mot message va chi danh dau published sau khi broker ack."""


class SqlAlchemyOutboxRelay:
    """Doc event chua publish theo batch va danh dau atomic trong session."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession], producer: MessageProducer, topic: str) -> None:
        self._session_factory = session_factory
        self._producer = producer
        self._topic = topic

    async def relay_once(self, batch_size: int = 50) -> int:
        """Publish toi da batch_size event; Kafka loi de nguyen event de lan sau retry."""

        import json

        published_count = 0
        async with self._session_factory() as session:
            statement = (
                select(ImageOptimizationOutboxRecord)
                .where(ImageOptimizationOutboxRecord.published_at.is_(None))
                .order_by(ImageOptimizationOutboxRecord.event_id.asc())
                .limit(batch_size)
            )
            records = (await session.execute(statement)).scalars().all()
            for record in records:
                try:
                    await self._producer.send_and_wait(
                        self._topic,
                        json.dumps(record.payload).encode("utf-8"),
                        key=str(record.aggregate_id).encode("utf-8"),
                    )
                except Exception:
                    record.attempts += 1
                    record.last_error = "KAFKA_PUBLISH_FAILED"
                    continue
                record.published_at = datetime.now(UTC)
                record.attempts += 1
                record.last_error = None
                published_count += 1
            await session.commit()
        return published_count
