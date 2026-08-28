"""Use case khôi phục ảnh gốc sau khi output AI đã được áp dụng.

Product Service giữ snapshot ảnh gốc và là nơi thực hiện mutation cuối cùng.
"""

from uuid import UUID

from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository, ProductMediaClient
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Điều phối rollback idempotent qua Product Service.
class RollbackImageOptimizationJob:
    """Chỉ chuyển ROLLED_BACK sau khi downstream khôi phục thành công."""

    # Nhận repository và product media port.
    def __init__(self, repository: ImageOptimizationJobRepository, product_media_client: ProductMediaClient | None) -> None:
        """Lưu dependency cho một request."""

        self._repository = repository
        self._product_media_client = product_media_client

    # Thực hiện rollback cho đúng owner và đúng job đã apply.
    async def execute(
        self,
        job_id: UUID,
        seller_owner_id: UUID,
        permissions: frozenset[str] = frozenset(),
        seller_email: str = "",
    ) -> ImageOptimizationJob:
        """Retry job đã ROLLED_BACK trả kết quả cũ mà không gọi downstream lại."""

        job = await self._repository.find_by_id(job_id, seller_owner_id)
        if job is None:
            raise LookupError("Optimization job not found")
        if job.status is ImageOptimizationStatus.ROLLED_BACK:
            return job
        if self._product_media_client is not None:
            if seller_email:
                await self._product_media_client.rollback_media(
                    seller_owner_id=seller_owner_id,
                    product_id=job.product_id,
                    job_id=job.job_id,
                    permissions=tuple(sorted(permissions)),
                    seller_email=seller_email,
                )
            else:
                await self._product_media_client.rollback_media(
                    seller_owner_id=seller_owner_id,
                    product_id=job.product_id,
                    job_id=job.job_id,
                    permissions=tuple(sorted(permissions)),
                )
        updated = job.transition(ImageOptimizationStatus.ROLLED_BACK).release_lease()
        await self._repository.save(updated)
        return updated
