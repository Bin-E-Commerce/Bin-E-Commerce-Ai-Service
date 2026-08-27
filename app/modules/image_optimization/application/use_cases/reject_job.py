"""Use case từ chối output tối ưu ảnh.

Việc cleanup được gọi sau khi lưu REJECTED; lỗi cleanup được trả về để composition
layer có thể retry, không bị nuốt bởi suppress(Exception).
"""

from uuid import UUID

from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository, MediaAssetClient
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Chuyển job sang REJECTED và yêu cầu Media Service dọn output của đúng job.
class RejectImageOptimizationJob:
    """Không bao giờ xóa source assets hoặc ảnh gốc của product."""

    # Nhận repository và media lifecycle port.
    def __init__(self, repository: ImageOptimizationJobRepository, media_client: MediaAssetClient | None) -> None:
        """Lưu dependency; test có thể bỏ media client để chỉ kiểm tra domain."""

        self._repository = repository
        self._media_client = media_client

    # Thực hiện idempotent cho job đã REJECTED.
    async def execute(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob:
        """Raise LookupError cho presentation map 404 và giữ lỗi cleanup để retry."""

        job = await self._repository.find_by_id(job_id, seller_owner_id)
        if job is None:
            raise LookupError("Optimization job not found")
        if job.status is ImageOptimizationStatus.REJECTED:
            # Retry cleanup ở request idempotent kế tiếp; không trả success giả khi media vẫn chưa được dọn.
            if self._media_client is not None:
                await self._media_client.cleanup_outputs(seller_owner_id=seller_owner_id, job_id=job.job_id)
            return job
        updated = job.transition(ImageOptimizationStatus.REJECTED).release_lease()
        await self._repository.save(updated)
        if self._media_client is not None:
            await self._media_client.cleanup_outputs(seller_owner_id=seller_owner_id, job_id=job.job_id)
        return updated
