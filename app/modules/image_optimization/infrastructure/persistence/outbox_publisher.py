"""Outbox publisher luu metadata event trong cung SQLAlchemy session voi aggregate job."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.image_optimization.domain.models import ImageOptimizationJob
from app.modules.image_optimization.infrastructure.persistence.models import ImageOptimizationOutboxRecord


class SqlAlchemyOptimizationOutboxPublisher:
    """Ghi event chua publish; publisher process rieng se doc va day Kafka sau commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Them mot event metadata, idempotent theo aggregate va event type trong transaction hien tai."""

        self._session.add(
            ImageOptimizationOutboxRecord(
                event_id=uuid4(),
                aggregate_id=job.job_id,
                event_type="ai.image-optimization.requested",
                payload={
                    "eventId": str(uuid4()),
                    "eventType": "ai.image-optimization.requested",
                    "schemaVersion": 1,
                    "jobId": str(job.job_id),
                    "productId": str(job.product_id),
                    "sellerOwnerId": str(job.seller_owner_id),
                    "sourceAssetIds": [str(value) for value in job.source_asset_ids],
                    "modes": [mode.value for mode in job.requested_modes],
                    "occurredAt": datetime.now(UTC).isoformat(),
                },
            )
        )
