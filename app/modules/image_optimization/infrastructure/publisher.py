"""Publisher local va event contract, san sang thay bang Kafka outbox publisher."""

from datetime import UTC, datetime
from typing import Any

from app.modules.image_optimization.domain.models import ImageOptimizationJob


class InMemoryOptimizationEventPublisher:
    """Giu event trong memory de worker local co the lay va test idempotency."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Tao event chi chua metadata, khong dua binary hay signed URL vao broker."""

        self.events.append(
            {
                "eventType": "ai.image-optimization.requested",
                "schemaVersion": 1,
                "jobId": str(job.job_id),
                "productId": str(job.product_id),
                "sellerOwnerId": str(job.seller_owner_id),
                "sourceAssetIds": [str(asset_id) for asset_id in job.source_asset_ids],
                "modes": [mode.value for mode in job.requested_modes],
                "occurredAt": datetime.now(UTC).isoformat(),
            }
        )
