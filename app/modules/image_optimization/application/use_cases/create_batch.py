"""Use case tạo batch tối ưu ảnh idempotent.

Use case xác minh ownership và source asset trước khi tiêu quota. File không biết
HTTP, SQLAlchemy, Kafka hoặc provider tạo ảnh cụ thể.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.core.errors import BackgroundConfigurationError, IdempotencyKeyReusedError, InvalidInputError
from app.modules.image_optimization.application.commands import CreateOptimizationJobsCommand
from app.modules.image_optimization.application.ports import (
    BackgroundDescriptionCipher,
    ImageOptimizationJobRepository,
    ImageOptimizationRateLimiter,
    OptimizationEventPublisher,
    ProductOwnerClient,
)
from app.modules.image_optimization.domain.models import ImageOptimizationBatch, ImageOptimizationJob


# Context đã được Product Service xác minh cho đúng một sản phẩm.
@dataclass(frozen=True)
class _ResolvedProduct:
    """Giữ version và source assets để bước ghi dữ liệu không gọi network lần nữa."""

    product_id: UUID
    expected_updated_at: datetime | None
    source_asset_ids: tuple[UUID, ...]


# Tạo một batch và các job con trong transaction do composition root quản lý.
class CreateImageOptimizationBatch:
    """Bảo vệ idempotency, ownership, asset policy và quota trước khi publish event."""

    # Nhận port thuần để unit test không cần PostgreSQL, Kafka hoặc Product Service thật.
    def __init__(
        self,
        *,
        repository: ImageOptimizationJobRepository,
        publisher: OptimizationEventPublisher,
        owner_client: ProductOwnerClient | None,
        rate_limiter: ImageOptimizationRateLimiter | None,
        rate_limit_requests: int,
        rate_limit_window_seconds: int,
        background_cipher: BackgroundDescriptionCipher | None,
        allow_unverified_memory_sources: bool,
    ) -> None:
        """Lưu dependency và buộc caller khai báo rõ khi dùng fallback memory cho test."""

        self._repository = repository
        self._publisher = publisher
        self._owner_client = owner_client
        self._rate_limiter = rate_limiter
        self._rate_limit_requests = rate_limit_requests
        self._rate_limit_window_seconds = rate_limit_window_seconds
        self._background_cipher = background_cipher
        self._allow_unverified_memory_sources = allow_unverified_memory_sources

    # Kiểm tra exact idempotency trước, xác minh tất cả sản phẩm rồi mới tiêu quota và ghi batch.
    async def execute(self, command: CreateOptimizationJobsCommand) -> tuple[str, tuple[ImageOptimizationJob, ...]]:
        """Trả batch cũ cho retry hợp lệ và 409 khi cùng key đại diện payload khác."""

        # Chan request sai truoc idempotency lookup de khong the dung key cu cho workflow nhieu doi tuong.
        if len(command.product_ids) != 1 or len(command.source_asset_ids) > 1:
            raise InvalidInputError()

        request_hash = command.request_hash()
        existing_batch = await self._repository.find_batch(command.seller_owner_id, command.idempotency_key)
        if existing_batch is not None:
            if existing_batch.request_hash != request_hash:
                raise IdempotencyKeyReusedError()
            return str(existing_batch.batch_id), await self._repository.find_jobs_by_batch(existing_batch.batch_id)

        if command.background_description and len(command.product_ids) != 1:
            raise InvalidInputError()
        if command.background_description and self._background_cipher is None:
            raise BackgroundConfigurationError()

        # Ownership và asset selection được resolve trước rate limit để request sai không làm mất lượt seller.
        # Tối đa 10 sản phẩm đã được validate ở presentation; resolve ownership song song để giảm tổng latency.
        # Mỗi coroutine vẫn gọi Product Service độc lập và không chia sẻ state mutable giữa các sản phẩm.
        # Request chi con mot san pham, sau khi resolve van bat buoc phai co dung mot source asset.
        resolved_products = tuple(
            await asyncio.gather(*(self._resolve_product(command, product_id) for product_id in command.product_ids))
        )
        if len(resolved_products) != 1 or len(resolved_products[0].source_asset_ids) != 1:
            raise InvalidInputError()
        if self._rate_limiter is not None:
            await self._rate_limiter.check(
                key=f"ai:image-optimization:{command.seller_owner_id}",
                limit=self._rate_limit_requests,
                window_seconds=self._rate_limit_window_seconds,
            )

        batch = ImageOptimizationBatch.create(
            seller_owner_id=command.seller_owner_id,
            idempotency_key=command.idempotency_key,
            request_hash=request_hash,
        )
        persisted_batch = await self._repository.save_batch(batch)
        if persisted_batch.batch_id != batch.batch_id:
            if persisted_batch.request_hash != request_hash:
                raise IdempotencyKeyReusedError()
            return str(persisted_batch.batch_id), await self._repository.find_jobs_by_batch(persisted_batch.batch_id)
        background_ciphertext = (
            self._background_cipher.encrypt(command.background_description)
            if command.background_description and self._background_cipher is not None
            else None
        )
        background_hash = (
            hashlib.sha256(command.background_description.encode("utf-8")).hexdigest() if command.background_description else None
        )

        jobs: list[ImageOptimizationJob] = []
        for resolved in resolved_products:
            job = ImageOptimizationJob.create(
                seller_owner_id=command.seller_owner_id,
                product_id=resolved.product_id,
                source_asset_ids=resolved.source_asset_ids,
                requested_modes=command.modes,
                # Job dùng identity nội bộ theo batch/product; client key chỉ nằm ở batch unique owner/key.
                idempotency_key=f"{batch.batch_id}:{resolved.product_id}",
                expected_product_updated_at=resolved.expected_updated_at,
                batch_id=batch.batch_id,
                request_hash=request_hash,
                background_preset=command.background_preset,
                background_description_ciphertext=background_ciphertext,
                background_description_hash=background_hash,
            )
            await self._repository.save(job)
            await self._publisher.publish_requested(job)
            jobs.append(job)
        return str(batch.batch_id), tuple(jobs)

    # Áp dụng chính xác sourceAssetPolicy và không tin asset IDs do browser tự gửi.
    async def _resolve_product(self, command: CreateOptimizationJobsCommand, product_id: UUID) -> _ResolvedProduct:
        """Trả source/version đã xác minh hoặc chỉ cho fallback khi runtime mode memory được khai báo."""

        if self._owner_client is None:
            if not self._allow_unverified_memory_sources:
                raise BackgroundConfigurationError()
            source_ids = command.source_asset_ids or (product_id,)
            return _ResolvedProduct(product_id, command.expected_product_updated_at, source_ids)

        expected_updated_at = await self._owner_client.assert_owned_and_get_updated_at(
            command.seller_owner_id,
            product_id,
            command.permissions,
            command.seller_email,
        )
        if command.source_asset_policy == "SELECTED_ASSETS":
            if not command.source_asset_ids:
                raise InvalidInputError()
            source_ids = await self._owner_client.get_product_asset_ids(
                command.seller_owner_id,
                product_id,
                command.source_asset_ids,
                command.permissions,
                command.seller_email,
            )
        elif command.source_asset_policy == "COVER_IMAGE":
            source_ids = (
                await self._owner_client.get_cover_asset_id(
                    command.seller_owner_id,
                    product_id,
                    command.permissions,
                    command.seller_email,
                ),
            )
        else:
            raise InvalidInputError()
        return _ResolvedProduct(product_id, expected_updated_at, source_ids)
