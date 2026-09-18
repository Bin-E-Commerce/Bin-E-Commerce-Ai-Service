"""Use case áp dụng các output ảnh seller đã chọn.

Use case đối chiếu asset IDs với aggregate và chỉ chuyển APPLIED sau khi Product
Service chấp nhận. Retry dùng cùng job ID để downstream có thể xử lý idempotent.
"""

from app.core.errors import OptimizationJobNotReadyError
from app.modules.image_optimization.application.contracts.commands import ApplyOptimizationOutputsCommand
from app.modules.image_optimization.application.ports import (
    ImageOptimizationJobRepository,
    OptimizationEventPublisher,
    ProductMediaClient,
    ProductOwnerClient,
)
from app.modules.image_optimization.domain.enums import ImageGenerationProfile, ImageOptimizationStatus
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError, InvalidOutputSelectionError
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob


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
        publisher: OptimizationEventPublisher | None = None,
        finalize_before_apply: bool = False,
        owner_client: ProductOwnerClient | None = None,
    ) -> None:
        """Chỉ cho phép thiếu downstream trong test/runtime memory được khai báo rõ."""

        self._repository = repository
        self._product_media_client = product_media_client
        self._allow_memory_without_downstream = allow_memory_without_downstream
        self._publisher = publisher
        self._finalize_before_apply = finalize_before_apply
        self._owner_client = owner_client

    # Xác minh lifecycle, optimistic version và selected assets trước side effect.
    async def execute(self, command: ApplyOptimizationOutputsCommand) -> ImageOptimizationJob:
        """Apply toàn bộ output khi images rỗng để giữ tương thích client cũ."""

        job = await self._repository.find_by_id(command.job_id, command.seller_owner_id)
        if job is None:
            raise OptimizationJobNotReadyError()
        if job.status is ImageOptimizationStatus.APPLIED:
            return job
        # Khi worker dang tao ban final, request lap lai chi xac nhan lan chay da duoc tiep nhan.
        # Khong goi Product Service hoac provider lan nua de tranh duplicate side effect va chi phi AI.
        if job.status is ImageOptimizationStatus.FINALIZING:
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

        # Production tạo ảnh medium trước; memory tests vẫn giữ contract apply cũ để không cần broker thật.
        if self._finalize_before_apply and job.generation_profile is ImageGenerationProfile.PREVIEW:
            try:
                finalizing = job.request_finalization(command.selected_asset_ids)
            except (InvalidJobTransitionError, InvalidOutputSelectionError) as error:
                raise OptimizationJobNotReadyError() from error
            await self._repository.save(finalizing)
            if self._publisher is None:
                raise OptimizationJobNotReadyError()
            await self._publisher.publish_requested(finalizing)
            return finalizing

        # Chỉ mốc analytics khi source asset vẫn là cover tại thời điểm apply; preview không tạo session.
        starts_impact_session = await self._resolve_impact_session_boundary(job, selected_outputs, command)

        if self._product_media_client is None:
            if not self._allow_memory_without_downstream:
                raise OptimizationJobNotReadyError()
        else:
            if command.seller_email:
                await self._product_media_client.apply_media(
                    seller_owner_id=command.seller_owner_id,
                    product_id=job.product_id,
                    job_id=job.job_id,
                    expected_product_updated_at=job.expected_product_updated_at,
                    assets=selected_outputs,
                    permissions=tuple(sorted(command.permissions)),
                    seller_email=command.seller_email,
                )
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
            updated = (
                job.transition(ImageOptimizationStatus.APPLIED)
                .release_lease()
                .mark_impact_session_boundary(starts_impact_session)
            )
        except InvalidJobTransitionError as error:
            raise OptimizationJobNotReadyError() from error
        await self._repository.save(updated)
        return updated

    # Xác định cover qua Product Service thay vì tin asset ID do browser gửi hoặc giá trị lúc tạo job.
    async def _resolve_impact_session_boundary(
        self,
        job: ImageOptimizationJob,
        selected_outputs: tuple[GeneratedAsset, ...],
        command: ApplyOptimizationOutputsCommand,
    ) -> bool:
        """Trả true khi output đang apply thay ảnh đại diện hiện tại của product."""

        if self._owner_client is None:
            # Memory runtime không có Product Service để xác minh cover; giữ behavior cũ cho test/local.
            return True
        cover_asset_id = await self._owner_client.get_cover_asset_id(
            command.seller_owner_id,
            job.product_id,
            command.permissions,
            command.seller_email,
        )
        return any(output.source_asset_id == cover_asset_id for output in selected_outputs)
