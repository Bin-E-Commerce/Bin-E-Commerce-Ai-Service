"""Relay outbox PostgreSQL sang Kafka với claim, backoff và dead-letter.

Relay không tạo event mới và không sửa payload. Nhiều instance dùng
`FOR UPDATE SKIP LOCKED` để không cùng publish một row trong một thời điểm.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from aiokafka.errors import KafkaError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.image_optimization.infrastructure.persistence.models import ImageOptimizationOutboxRecord


# Contract nhỏ với Kafka producer để integration test dùng fake có cùng API.
class MessageProducer(Protocol):
    """Chỉ đánh dấu published sau khi broker xác nhận send_and_wait."""

    # Gửi một message có key aggregate để giữ ordering trong partition.
    async def send_and_wait(self, topic: str, value: bytes, key: bytes | None = None) -> object:
        """Trả metadata broker hoặc raise KafkaError."""


# Snapshot event trước khi commit claim để không giữ row lock trong lúc chờ Kafka.
@dataclass(frozen=True)
class _ClaimedOutboxEvent:
    """Metadata tối thiểu cần publish và cập nhật trạng thái sau đó."""

    event_id: object
    aggregate_id: object
    payload: dict[str, object]


# Claim và relay một batch outbox có giới hạn retry.
class SqlAlchemyOutboxRelay:
    """Không nuốt lỗi ngoài Kafka và không lưu exception message nhạy cảm."""

    # Nhận session factory, producer và policy retry từ composition root.
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        producer: MessageProducer,
        topic: str,
        *,
        dlq_topic: str | None = None,
        max_attempts: int = 8,
        relay_id: str | None = None,
    ) -> None:
        """Mỗi process có relay ID riêng để audit claim mà không chứa host/user data."""

        self._session_factory = session_factory
        self._producer = producer
        self._topic = topic
        self._dlq_topic = dlq_topic
        self._max_attempts = max(1, max_attempts)
        self._relay_id = relay_id or f"outbox-{uuid4()}"

    # Claim row trong transaction ngắn, sau đó publish ngoài transaction để không chặn API ghi job.
    async def relay_once(self, batch_size: int = 50) -> int:
        """Event Kafka có thể at-least-once; worker vẫn phải idempotent theo job/event ID."""

        now = datetime.now(UTC)
        claimed_events: tuple[_ClaimedOutboxEvent, ...]
        async with self._session_factory() as session:
            statement = (
                select(ImageOptimizationOutboxRecord)
                .where(
                    ImageOptimizationOutboxRecord.published_at.is_(None),
                    ImageOptimizationOutboxRecord.dead_lettered_at.is_(None),
                    or_(
                        ImageOptimizationOutboxRecord.next_attempt_at.is_(None),
                        ImageOptimizationOutboxRecord.next_attempt_at <= now,
                    ),
                    or_(
                        ImageOptimizationOutboxRecord.locked_by.is_(None),
                        ImageOptimizationOutboxRecord.locked_at.is_(None),
                        ImageOptimizationOutboxRecord.locked_at < now - timedelta(minutes=5),
                    ),
                )
                .order_by(ImageOptimizationOutboxRecord.event_id.asc())
                .limit(max(1, batch_size))
                .with_for_update(skip_locked=True)
            )
            records = (await session.execute(statement)).scalars().all()
            for record in records:
                record.locked_by = self._relay_id
                record.locked_at = now
            claimed_events = tuple(
                _ClaimedOutboxEvent(record.event_id, record.aggregate_id, record.payload) for record in records
            )
            await session.commit()

        published_count = 0
        for event in claimed_events:
            try:
                await self._producer.send_and_wait(
                    self._topic,
                    json.dumps(event.payload, separators=(",", ":")).encode("utf-8"),
                    key=str(event.aggregate_id).encode("utf-8"),
                )
            except KafkaError:
                await self._mark_retry_or_dlq(event.event_id, now)
                continue
            await self._mark_published(event.event_id, now)
            published_count += 1
        return published_count

    # Đánh dấu publish bằng transaction riêng; crash giữa Kafka và bước này chỉ tạo duplicate at-least-once.
    async def _mark_published(self, event_id: object, now: datetime) -> None:
        """Chỉ relay đang giữ lock mới được hoàn tất event để instance khác không ghi đè trạng thái."""

        async with self._session_factory() as session:
            record = (
                await session.execute(
                    select(ImageOptimizationOutboxRecord)
                    .where(
                        ImageOptimizationOutboxRecord.event_id == event_id,
                        ImageOptimizationOutboxRecord.locked_by == self._relay_id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if record is None:
                return
            record.published_at = now
            record.attempts += 1
            record.last_error = None
            record.next_attempt_at = None
            record.locked_by = None
            record.locked_at = None
            await session.commit()

    # Tăng attempt với exponential backoff hoặc gửi metadata sang DLQ khi hết retry.
    async def _mark_retry_or_dlq(self, event_id: object, now: datetime) -> None:
        """Không đưa raw exception vào database hoặc DLQ."""

        async with self._session_factory() as session:
            record = (
                await session.execute(
                    select(ImageOptimizationOutboxRecord)
                    .where(
                        ImageOptimizationOutboxRecord.event_id == event_id,
                        ImageOptimizationOutboxRecord.locked_by == self._relay_id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if record is None:
                return
            record.attempts += 1
            record.last_error = "KAFKA_PUBLISH_FAILED"
            record.locked_by = None
            record.locked_at = None
            if record.attempts < self._max_attempts:
                delay_seconds = min(300, 2 ** min(record.attempts, 8))
                record.next_attempt_at = now + timedelta(seconds=delay_seconds)
            else:
                if self._dlq_topic is not None:
                    try:
                        # DLQ là side effect hiếm; chỉ chốt dead-letter sau khi broker xác nhận đã nhận.
                        await self._producer.send_and_wait(
                            self._dlq_topic,
                            json.dumps(record.payload, separators=(",", ":")).encode("utf-8"),
                            key=str(record.aggregate_id).encode("utf-8"),
                        )
                    except KafkaError:
                        record.next_attempt_at = now + timedelta(minutes=5)
                    else:
                        record.dead_lettered_at = now
                        record.next_attempt_at = None
                else:
                    record.dead_lettered_at = now
                    record.next_attempt_at = None
            await session.commit()
