"""Publisher local va event contract, san sang thay bang Kafka outbox publisher."""

from typing import Any

from app.modules.image_optimization.application.events import ImageOptimizationRequestedEvent
from app.modules.image_optimization.domain.models import ImageOptimizationJob


class InMemoryOptimizationEventPublisher:
    """Giu event trong memory de worker local co the lay va test idempotency."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Tao event chi chua metadata, khong dua binary hay signed URL vao broker."""

        self.events.append(ImageOptimizationRequestedEvent.from_job(job).to_payload())
