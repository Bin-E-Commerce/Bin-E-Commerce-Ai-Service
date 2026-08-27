"""Use case tổng hợp metric job tối ưu ảnh cho seller.

Các phép đếm được giao cho repository để production dùng SQL COUNT thay vì tải toàn bộ row.
"""

from uuid import UUID

from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus


# Đọc các metric chỉ thuộc seller hiện tại.
class GetImageOptimizationOverview:
    """Trả primitive an toàn cho presentation, không tạo số lượt xem/lượt bán giả."""

    # Nhận repository query port.
    def __init__(self, repository: ImageOptimizationJobRepository) -> None:
        """Lưu dependency cho một request."""

        self._repository = repository

    # Chạy các truy vấn đếm độc lập và giữ metric chưa tích hợp ở giá trị None.
    async def execute(self, seller_owner_id: UUID) -> dict[str, int | None]:
        """Không suy diễn Product Service analytics trong AI Service."""

        return {
            "optimized_products": await self._repository.count_applied(seller_owner_id),
            "total_views": None,
            "total_sold": None,
            "pending_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.PENDING),
            "failed_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.FAILED),
        }
