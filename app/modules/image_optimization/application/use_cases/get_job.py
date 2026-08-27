"""Use case đọc một job theo seller owner.

File không map HTTP 404; presentation chịu trách nhiệm chuyển kết quả None thành response.
"""

from uuid import UUID

from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Đọc job bằng owner filter để không lộ tài nguyên giữa các seller.
class GetImageOptimizationJob:
    """Use case query nhỏ với duy nhất một hàm execute."""

    # Nhận repository port để query không phụ thuộc SQLAlchemy.
    def __init__(self, repository: ImageOptimizationJobRepository) -> None:
        """Lưu repository cho vòng đời của request."""

        self._repository = repository

    # Trả job thuộc seller hoặc None khi không tồn tại/không thuộc quyền sở hữu.
    async def execute(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob | None:
        """Không phân biệt not-found và cross-owner để tránh lộ dữ liệu."""

        return await self._repository.find_by_id(job_id, seller_owner_id)
