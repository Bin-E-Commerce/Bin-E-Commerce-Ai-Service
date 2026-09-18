"""Use case tổng hợp metric job tối ưu ảnh cho seller.

Các phép đếm được giao cho repository để production dùng SQL COUNT thay vì tải toàn bộ row.
"""

from uuid import UUID

from app.modules.image_optimization.application.policies.quota import image_optimization_rate_limit_key
from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository, ImageOptimizationRateLimiter
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus


# Đọc các metric chỉ thuộc seller hiện tại.
class GetImageOptimizationOverview:
    """Trả primitive an toàn cho presentation, không tạo số lượt xem/lượt bán giả."""

    # Nhận repository query port.
    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        rate_limiter: ImageOptimizationRateLimiter | None = None,
        rate_limit_requests: int = 3,
        rate_limit_window_seconds: int = 3600,
    ) -> None:
        """Lưu dependency cho một request."""

        self._repository = repository
        self._rate_limiter = rate_limiter
        self._rate_limit_requests = rate_limit_requests
        self._rate_limit_window_seconds = rate_limit_window_seconds

    # Chạy các truy vấn đếm độc lập và giữ metric chưa tích hợp ở giá trị None.
    async def execute(self, seller_owner_id: UUID) -> dict[str, object]:
        """Không suy diễn Product Service analytics trong AI Service."""

        if self._rate_limiter is None:
            ai_usage = {
                "enabled": False,
                "limit": None,
                "used": 0,
                "remaining": None,
            }
        else:
            used = await self._rate_limiter.get_usage(
                image_optimization_rate_limit_key(seller_owner_id),
                self._rate_limit_window_seconds,
            )
            ai_usage = {
                "enabled": True,
                "limit": self._rate_limit_requests,
                "used": used,
                "remaining": max(0, self._rate_limit_requests - used),
            }

        return {
            "optimized_products": await self._repository.count_applied(seller_owner_id),
            "total_views": None,
            "total_sold": None,
            "pending_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.PENDING),
            "failed_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.FAILED),
            "ai_usage": ai_usage,
        }
