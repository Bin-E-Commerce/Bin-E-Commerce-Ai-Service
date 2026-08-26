"""Repository memory dung cho local/test; production co the thay bang SQLAlchemy adapter."""

from uuid import UUID

from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob


class InMemoryImageOptimizationJobRepository:
    """Luu job trong process de API local chay ngay khi chua bat database."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, ImageOptimizationJob] = {}

    async def save(self, job: ImageOptimizationJob) -> None:
        """Ghi de aggregate theo ID, phu hop cho test state transition."""

        self._jobs[job.job_id] = job

    async def find_by_id(self, job_id: UUID, seller_owner_id: UUID | None = None) -> ImageOptimizationJob | None:
        """Doc job va chan seller khac doc duoc job cua nhau."""

        job = self._jobs.get(job_id)
        if job is None or (seller_owner_id is not None and job.seller_owner_id != seller_owner_id):
            return None
        return job

    async def find_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationJob | None:
        """Tim request cu trong cung seller de retry idempotent."""

        return next(
            (
                job
                for job in self._jobs.values()
                if job.seller_owner_id == seller_owner_id
                and (job.idempotency_key == idempotency_key or job.idempotency_key.startswith(f"{idempotency_key}:"))
            ),
            None,
        )

    async def count_applied(self, seller_owner_id: UUID) -> int:
        """Dem job da apply ma khong suy dien metric tu UI."""

        return sum(
            1
            for job in self._jobs.values()
            if job.seller_owner_id == seller_owner_id and job.status is ImageOptimizationStatus.APPLIED
        )

    async def find_batch_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> tuple[ImageOptimizationJob, ...]:
        """Lay tat ca job co prefix idempotency batch theo thu tu tao."""

        return tuple(
            sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.seller_owner_id == seller_owner_id
                    and (job.idempotency_key == idempotency_key or job.idempotency_key.startswith(f"{idempotency_key}:"))
                ),
                key=lambda job: job.created_at,
            )
        )

    async def count_status(self, seller_owner_id: UUID, status: ImageOptimizationStatus) -> int:
        """Dem state tu memory cho dashboard/test."""

        return sum(1 for job in self._jobs.values() if job.seller_owner_id == seller_owner_id and job.status is status)
