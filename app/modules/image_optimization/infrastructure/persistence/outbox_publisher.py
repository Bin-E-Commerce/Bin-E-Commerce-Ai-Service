"""Outbox publisher luu metadata event trong cung SQLAlchemy session voi aggregate job."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.image_optimization.application.events import ImageOptimizationRequestedEvent
from app.modules.image_optimization.domain.models import ImageOptimizationJob
from app.modules.image_optimization.infrastructure.persistence.models import ImageOptimizationOutboxRecord


class SqlAlchemyOptimizationOutboxPublisher:
    """Ghi event chua publish; publisher process rieng se doc va day Kafka sau commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Them mot event metadata, idempotent theo aggregate va event type trong transaction hien tai."""

        event_id = uuid4()
        event = ImageOptimizationRequestedEvent.from_job(job, event_id=event_id)
        self._session.add(
            ImageOptimizationOutboxRecord(
                event_id=event_id,
                aggregate_id=job.job_id,
                event_type=event.event_type,
                payload=event.to_payload(),
            )
        )
