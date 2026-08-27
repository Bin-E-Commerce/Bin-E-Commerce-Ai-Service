"""Facade tương thích tạm thời cho các use case tối ưu ảnh đã được tách riêng.

Router cũ vẫn phụ thuộc class này để không đổi dependency override công khai. Mọi
logic nghiệp vụ đã nằm trong class use case có một hàm `execute` duy nhất.
"""

from datetime import datetime
from uuid import UUID

from app.modules.image_optimization.application.commands import (
    ApplyOptimizationOutputsCommand,
    CreateOptimizationJobsCommand,
)
from app.modules.image_optimization.application.ports import (
    BackgroundDescriptionCipher,
    ImageOptimizationJobRepository,
    ImageOptimizationRateLimiter,
    MediaAssetClient,
    OptimizationEventPublisher,
    ProductMediaClient,
    ProductOwnerClient,
)
from app.modules.image_optimization.application.use_cases import (
    ApplyImageOptimizationOutputs,
    CreateImageOptimizationBatch,
    GetImageOptimizationJob,
    GetImageOptimizationOverview,
    RejectImageOptimizationJob,
    RollbackImageOptimizationJob,
)
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Gom các use case đã tách để giữ wiring hiện tại trong lúc presentation chuyển dần sang inject từng use case.
class ImageOptimizationApplicationService:
    """Không chứa business branching; mỗi method chỉ chuyển tiếp sang một use case."""

    # Khởi tạo từng use case từ các port đã được composition root cung cấp.
    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        publisher: OptimizationEventPublisher,
        owner_client: ProductOwnerClient | None = None,
        product_media_client: ProductMediaClient | None = None,
        media_asset_client: MediaAssetClient | None = None,
        rate_limiter: ImageOptimizationRateLimiter | None = None,
        rate_limit_requests: int = 3,
        rate_limit_window_seconds: int = 3600,
        background_cipher: BackgroundDescriptionCipher | None = None,
        *,
        allow_memory_adapters: bool = True,
    ) -> None:
        """Chỉ cho phép downstream thiếu khi runtime mode memory được truyền rõ ràng."""

        self._create = CreateImageOptimizationBatch(
            repository=repository,
            publisher=publisher,
            owner_client=owner_client,
            rate_limiter=rate_limiter,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window_seconds=rate_limit_window_seconds,
            background_cipher=background_cipher,
            allow_unverified_memory_sources=allow_memory_adapters,
        )
        self._get = GetImageOptimizationJob(repository)
        self._overview = GetImageOptimizationOverview(repository)
        self._apply = ApplyImageOptimizationOutputs(
            repository,
            product_media_client,
            allow_memory_without_downstream=allow_memory_adapters,
        )
        self._reject = RejectImageOptimizationJob(repository, media_asset_client)
        self._rollback = RollbackImageOptimizationJob(repository, product_media_client)

    # Chuyển command tạo batch sang use case độc lập.
    async def create_jobs(self, command: CreateOptimizationJobsCommand) -> tuple[str, tuple[ImageOptimizationJob, ...]]:
        """Giữ method cũ để router và test không cần đổi đồng thời."""

        return await self._create.execute(command)

    # Chuyển query job sang use case query độc lập.
    async def get_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob | None:
        """Không chứa mapping HTTP hoặc persistence logic."""

        return await self._get.execute(job_id, seller_owner_id)

    # Chuyển query overview sang use case query độc lập.
    async def get_overview(self, seller_owner_id: UUID) -> dict[str, int | None]:
        """Để repository tự thực hiện COUNT hiệu quả."""

        return await self._overview.execute(seller_owner_id)

    # Chuyển reject sang use case lifecycle độc lập.
    async def reject_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob:
        """Không nuốt lỗi cleanup từ Media Service."""

        return await self._reject.execute(job_id, seller_owner_id)

    # Chuyển rollback sang use case lifecycle độc lập.
    async def rollback_job(
        self,
        job_id: UUID,
        seller_owner_id: UUID,
        permissions: frozenset[str] = frozenset(),
    ) -> ImageOptimizationJob:
        """Product Service vẫn là source of truth cho snapshot ảnh gốc."""

        return await self._rollback.execute(job_id, seller_owner_id, permissions)

    # Chuyển apply sang use case và truyền đúng asset IDs seller chọn cùng permission đã xác thực.
    async def apply_job(
        self,
        job_id: UUID,
        seller_owner_id: UUID,
        *,
        expected_product_updated_at: datetime | None = None,
        selected_asset_ids: tuple[UUID, ...] = (),
        permissions: frozenset[str] = frozenset(),
    ) -> ImageOptimizationJob:
        """Fallback version từ job chỉ phục vụ call cũ; router mới luôn truyền version request."""

        if expected_product_updated_at is None:
            job = await self._get.execute(job_id, seller_owner_id)
            if job is None or job.expected_product_updated_at is None:
                from app.core.errors import OptimizationJobNotReadyError

                raise OptimizationJobNotReadyError()
            expected_product_updated_at = job.expected_product_updated_at
        return await self._apply.execute(
            ApplyOptimizationOutputsCommand(
                job_id=job_id,
                seller_owner_id=seller_owner_id,
                expected_product_updated_at=expected_product_updated_at,
                selected_asset_ids=selected_asset_ids,
                permissions=permissions,
            )
        )
