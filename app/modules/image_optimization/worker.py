"""Worker entrypoint xu ly Kafka event va cap nhat job sau khi provider tao output."""

from collections.abc import Awaitable, Callable
from uuid import UUID

from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.ports import ImageOptimizationJobRepository


class ImageOptimizationWorker:
    """Orchestrator idempotent; Kafka adapter chi can goi process_event."""

    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        process_job: Callable[[UUID], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._process_job = process_job

    async def process_event(self, job_id: UUID) -> None:
        """Nhan job mot lan, bo qua job da terminal de Kafka redelivery khong tao output trung."""

        job = await self._repository.find_by_id(job_id)
        if job is None or job.status in {
            ImageOptimizationStatus.APPLIED,
            ImageOptimizationStatus.REJECTED,
            ImageOptimizationStatus.ROLLED_BACK,
        }:
            return
        await self._process_job(job_id)
