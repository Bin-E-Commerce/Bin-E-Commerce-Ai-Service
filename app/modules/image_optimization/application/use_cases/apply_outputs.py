"""Use case áp dụng các output ảnh seller đã chọn.

Use case đối chiếu asset IDs với aggregate và chỉ chuyển APPLIED sau khi Product
Service chấp nhận. Retry dùng cùng job ID để downstream có thể xử lý idempotent.
"""

from app.core.errors import OptimizationJobNotReadyError
from app.modules.image_optimization.application.commands import ApplyOptimizationOutputsCommand
from app.modules.image_optimization.application.ports import ImageOptimizationJobRepository, ProductMediaClient
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError, InvalidOutputSelectionError
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Điều phối việc apply output đã xác minh sang Product Service.
class ApplyImageOptimizationOutputs:
    """Không tin URL/mode từ browser và không tự tổng hợp permission nội bộ."""

    # Nhận repository và downstream port; production bắt buộc có Product Service client.
    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        product_media_client: ProductMediaClient | None,
        *,
        allow_memory_without_downstream: bool,
    ) -> None:
        """Chỉ cho phép thiếu downstream trong test/runtime memory được khai báo rõ."""

        self._repository = repository
        self._product_media_client = product_media_client
        self._allow_memory_without_downstream = allow_memory_without_downstream

    # Xác minh lifecycle, optimistic version và selected assets trước side effect.
    async def execute(self, command: ApplyOptimizationOutputsCommand) -> ImageOptimizationJob:
        """Apply toàn bộ output khi images rỗng để giữ tương thích client cũ."""

        job = await self._repository.find_by_id(command.job_id, command.seller_owner_id)
        if job is None:
            raise OptimizationJobNotReadyError()
        if job.status is ImageOptimizationStatus.APPLIED:
            return job
        if job.status not in {ImageOptimizationStatus.REVIEW_REQUIRED, ImageOptimizationStatus.SUCCEEDED}:
            raise OptimizationJobNotReadyError()
        if job.expected_product_updated_at is None or command.expected_product_updated_at != job.expected_product_updated_at:
            raise OptimizationJobNotReadyError()
        try:
            selected_outputs = job.select_outputs(command.selected_asset_ids)
        except InvalidOutputSelectionError as error:
            raise OptimizationJobNotReadyError() from error
        if not selected_outputs or any(not output.public_url for output in selected_outputs):
            raise OptimizationJobNotReadyError()

        if self._product_media_client is None:
            if not self._allow_memory_without_downstream:
                raise OptimizationJobNotReadyError()
        else:
            await self._product_media_client.apply_media(
                seller_owner_id=command.seller_owner_id,
                product_id=job.product_id,
                job_id=job.job_id,
                expected_product_updated_at=job.expected_product_updated_at,
                assets=selected_outputs,
                permissions=tuple(sorted(command.permissions)),
            )
        try:
            updated = job.transition(ImageOptimizationStatus.APPLIED).release_lease()
        except InvalidJobTransitionError as error:
            raise OptimizationJobNotReadyError() from error
        await self._repository.save(updated)
        return updated
