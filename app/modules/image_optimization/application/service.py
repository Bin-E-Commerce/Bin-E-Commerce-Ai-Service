"""Điều phối tạo job tối ưu ảnh, xác minh ownership, chống trùng và bảo vệ mô tả bối cảnh trước persistence."""

import hashlib
from contextlib import suppress
from uuid import UUID

from app.core.errors import BackgroundConfigurationError, InvalidInputError
from app.modules.image_optimization.application.commands import CreateOptimizationJobsCommand
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob
from app.modules.image_optimization.domain.ports import (
    BackgroundDescriptionCipher,
    ImageOptimizationJobRepository,
    ImageOptimizationRateLimiter,
    MediaAssetClient,
    OptimizationEventPublisher,
    ProductMediaClient,
    ProductOwnerClient,
)


class ImageOptimizationApplicationService:
    """Use case chinh cho batch create va truy van overview cua seller."""

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
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._owner_client = owner_client
        self._product_media_client = product_media_client
        self._media_asset_client = media_asset_client
        self._rate_limiter = rate_limiter
        self._rate_limit_requests = rate_limit_requests
        self._rate_limit_window_seconds = rate_limit_window_seconds
        self._background_cipher = background_cipher

    async def create_jobs(self, command: CreateOptimizationJobsCommand) -> tuple[str, tuple[ImageOptimizationJob, ...]]:
        """Tao job idempotent cho tung product, kiem tra ownership truoc khi tieu quota va publish event.

        Moi product la mot aggregate rieng de worker co the retry doc lap. Khi client gui lai cung
        Idempotency-Key, service tra lai job cu thay vi goi Kafka them lan nua. Neu chua co Product
        Service client trong local, boundary van cho phep chay demo; production wiring bat buoc client.
        """

        # Mô tả tùy chỉnh là request trả phí nên chỉ hỗ trợ một sản phẩm để kết quả đúng ý và tránh phát tán cùng một prompt.
        if command.background_description and len(command.product_ids) != 1:
            raise InvalidInputError()
        if command.background_description and self._background_cipher is None:
            # Không ghi mô tả nền dạng rõ vào database; thiếu khóa mã hóa phải dừng trước khi tạo job.
            raise BackgroundConfigurationError()
        existing = await self._repository.find_batch_by_idempotency(command.seller_owner_id, command.idempotency_key)
        if existing:
            return command.idempotency_key, existing
        if self._rate_limiter is not None:
            await self._rate_limiter.check(
                key=f"ai:image-optimization:{command.seller_owner_id}",
                limit=self._rate_limit_requests,
                window_seconds=self._rate_limit_window_seconds,
            )

        jobs: list[ImageOptimizationJob] = []
        for product_id in command.product_ids:
            expected_updated_at = command.expected_product_updated_at
            if self._owner_client is not None:
                expected_updated_at = await self._owner_client.assert_owned_and_get_updated_at(
                    command.seller_owner_id, product_id
                )
            source_asset_ids = command.source_asset_ids
            if self._owner_client is not None:
                if source_asset_ids:
                    source_asset_ids = await self._owner_client.get_product_asset_ids(
                        command.seller_owner_id, product_id, source_asset_ids
                    )
                else:
                    source_asset_ids = (await self._owner_client.get_cover_asset_id(command.seller_owner_id, product_id),)
            elif not source_asset_ids:
                # Fallback chỉ phục vụ adapter memory/test; production wiring luôn xác minh asset qua Product Service.
                source_asset_ids = (product_id,)
            # Cần lưu ciphertext vì worker chạy ở process khác; hash hỗ trợ đối chiếu/debug mà không tiết lộ nội dung seller viết.
            background_ciphertext = (
                self._background_cipher.encrypt(command.background_description)
                if command.background_description and self._background_cipher is not None
                else None
            )
            background_hash = (
                hashlib.sha256(command.background_description.encode("utf-8")).hexdigest()
                if command.background_description
                else None
            )
            job = ImageOptimizationJob.create(
                seller_owner_id=command.seller_owner_id,
                product_id=product_id,
                source_asset_ids=source_asset_ids,
                requested_modes=command.modes,
                idempotency_key=f"{command.idempotency_key}:{product_id}",
                expected_product_updated_at=expected_updated_at,
                background_preset=command.background_preset,
                background_description_ciphertext=background_ciphertext,
                background_description_hash=background_hash,
            )
            await self._repository.save(job)
            await self._publisher.publish_requested(job)
            jobs.append(job)
        return command.idempotency_key, tuple(jobs)

    async def get_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob | None:
        """Doc job theo seller de presentation khong can biet cach luu persistence."""

        return await self._repository.find_by_id(job_id, seller_owner_id)

    async def get_overview(self, seller_owner_id: UUID) -> dict[str, int | None]:
        """Tra metric khong co du lieu ao; frontend se hien thi dash khi value chua co."""

        return {
            "optimized_products": await self._repository.count_applied(seller_owner_id),
            "total_views": None,
            "total_sold": None,
            "pending_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.PENDING),
            "failed_jobs": await self._repository.count_status(seller_owner_id, ImageOptimizationStatus.FAILED),
        }

    async def reject_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob:
        """Danh dau output bi seller tu choi de Media Service cleanup asset AI sau retention."""

        job = await self._repository.find_by_id(job_id, seller_owner_id)
        if job is None:
            raise LookupError("Optimization job not found")
        if job.status is ImageOptimizationStatus.REJECTED:
            return job
        updated = job.transition(ImageOptimizationStatus.REJECTED)
        await self._repository.save(updated)
        if self._media_asset_client is not None:
            # Cleanup co the retry theo retention; reject khong duoc rollback chi vi S3 tam thoi loi.
            with suppress(Exception):
                await self._media_asset_client.cleanup_outputs(seller_owner_id=seller_owner_id, job_id=job.job_id)
        return updated

    async def rollback_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob:
        """Danh dau rollback idempotent sau khi Product Service da phuc hoi snapshot anh goc."""

        job = await self._repository.find_by_id(job_id, seller_owner_id)
        if job is None:
            raise LookupError("Optimization job not found")
        if job.status is ImageOptimizationStatus.ROLLED_BACK:
            return job
        updated = job.transition(ImageOptimizationStatus.ROLLED_BACK)
        if self._product_media_client is not None:
            await self._product_media_client.rollback_media(
                seller_owner_id=seller_owner_id,
                product_id=job.product_id,
                job_id=job.job_id,
            )
        await self._repository.save(updated)
        return updated

    async def apply_job(self, job_id: UUID, seller_owner_id: UUID) -> ImageOptimizationJob:
        """Chuyen job sang APPLIED qua application boundary sau khi output da san sang."""

        job = await self._repository.find_by_id(job_id, seller_owner_id)
        if job is None:
            raise LookupError("Optimization job not found")
        if job.status is ImageOptimizationStatus.APPLIED:
            return job
        if not job.generated_assets and not job.generated_asset_ids:
            raise ValueError("Optimization output is not ready")
        if self._product_media_client is not None and not any(asset.public_url for asset in job.generated_assets):
            raise ValueError("Optimization output has no public media URL")
        updated = job.transition(ImageOptimizationStatus.APPLIED)
        if self._product_media_client is not None:
            await self._product_media_client.apply_media(
                seller_owner_id=seller_owner_id,
                product_id=job.product_id,
                job_id=job.job_id,
                expected_product_updated_at=job.expected_product_updated_at,
                assets=job.generated_assets,
                permissions=("seller.ai.image_optimization.apply",),
            )
        await self._repository.save(updated)
        return updated
