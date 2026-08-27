"""Repository memory dành riêng cho test và runtime mode `memory`.

Adapter mô phỏng idempotency và claim concurrency bằng lock. Production không
được tự động fallback sang adapter này khi PostgreSQL thiếu cấu hình.
"""

import asyncio
from uuid import UUID

from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError
from app.modules.image_optimization.domain.models import ImageOptimizationBatch, ImageOptimizationJob


# Lưu aggregate trong process để test domain/use case không cần PostgreSQL.
class InMemoryImageOptimizationJobRepository:
    """Cung cấp cùng contract với SQL repository và bảo vệ claim bằng asyncio lock."""

    # Khởi tạo kho riêng cho batch, job và một lock bao quanh thao tác atomic.
    def __init__(self) -> None:
        """Không dùng global state để test không rò dữ liệu qua case khác."""

        self._jobs: dict[UUID, ImageOptimizationJob] = {}
        self._batches: dict[UUID, ImageOptimizationBatch] = {}
        self._batch_index: dict[tuple[UUID, str], UUID] = {}
        self._lock = asyncio.Lock()

    # Ghi aggregate theo ID; version được domain tăng ở mỗi mutation.
    async def save(self, job: ImageOptimizationJob) -> None:
        """Lưu snapshot mới cho test state transition."""

        async with self._lock:
            self._jobs[job.job_id] = job

    # Ghi batch và enforce unique seller/idempotency key giống PostgreSQL.
    async def save_batch(self, batch: ImageOptimizationBatch) -> ImageOptimizationBatch:
        """Từ chối ghi đè batch khác khi key đã tồn tại."""

        async with self._lock:
            key = (batch.seller_owner_id, batch.idempotency_key)
            existing_id = self._batch_index.get(key)
            if existing_id is not None and existing_id != batch.batch_id:
                return self._batches[existing_id]
            self._batches[batch.batch_id] = batch
            self._batch_index[key] = batch.batch_id
            return batch

    # Tìm batch bằng exact key để `_` và `%` không có semantics wildcard.
    async def find_batch(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationBatch | None:
        """Trả batch đúng seller hoặc None."""

        batch_id = self._batch_index.get((seller_owner_id, idempotency_key))
        return self._batches.get(batch_id) if batch_id else None

    # Đọc job theo batch ID nội bộ thay vì ghép prefix idempotency.
    async def find_jobs_by_batch(self, batch_id: UUID) -> tuple[ImageOptimizationJob, ...]:
        """Sắp xếp theo thời gian tạo để response batch ổn định."""

        return tuple(sorted((job for job in self._jobs.values() if job.batch_id == batch_id), key=lambda item: item.created_at))

    # Claim trong lock để hai coroutine không cùng nhận được quyền xử lý một job.
    async def claim_for_processing(
        self,
        job_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> ImageOptimizationJob | None:
        """Trả None nếu job không tồn tại, terminal hoặc đang có lease còn hiệu lực."""

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            try:
                claimed = job.claim(worker_id=worker_id, lease_seconds=lease_seconds)
            except InvalidJobTransitionError:
                return None
            self._jobs[job_id] = claimed
            return claimed

    # Đọc job và chặn seller khác truy cập aggregate.
    async def find_by_id(self, job_id: UUID, seller_owner_id: UUID | None = None) -> ImageOptimizationJob | None:
        """Trả None thay vì làm lộ việc job của seller khác tồn tại."""

        job = self._jobs.get(job_id)
        if job is None or (seller_owner_id is not None and job.seller_owner_id != seller_owner_id):
            return None
        return job

    # Giữ method legacy cho adapter cũ nhưng dùng exact key thông qua batch index.
    async def find_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationJob | None:
        """Trả job đầu tiên của batch chính xác, không dùng startswith."""

        batch = await self.find_batch(seller_owner_id, idempotency_key)
        if batch is None:
            return None
        jobs = await self.find_jobs_by_batch(batch.batch_id)
        return jobs[0] if jobs else None

    # Giữ contract cũ trong thời gian router chuyển sang batch ID.
    async def find_batch_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> tuple[ImageOptimizationJob, ...]:
        """Trả job của batch exact-key hoặc tuple rỗng."""

        batch = await self.find_batch(seller_owner_id, idempotency_key)
        return await self.find_jobs_by_batch(batch.batch_id) if batch else ()

    # Đếm job APPLIED trực tiếp trên collection memory.
    async def count_applied(self, seller_owner_id: UUID) -> int:
        """Phục vụ metric dashboard trong local/test."""

        return sum(
            1
            for job in self._jobs.values()
            if job.seller_owner_id == seller_owner_id and job.status is ImageOptimizationStatus.APPLIED
        )

    # Đếm trạng thái cụ thể cho dashboard local/test.
    async def count_status(self, seller_owner_id: UUID, status: ImageOptimizationStatus) -> int:
        """Không tạo số liệu giả khi repository rỗng."""

        return sum(1 for job in self._jobs.values() if job.seller_owner_id == seller_owner_id and job.status is status)
