"""Mô hình domain cho batch, job và output tối ưu ảnh.

File này chỉ chứa trạng thái và quy tắc nghiệp vụ thuần Python. File không được
biết FastAPI, SQLAlchemy, Kafka, Media Service hoặc provider AI cụ thể.
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.modules.image_optimization.domain.enums import (
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
    LifestyleBackgroundPreset,
)
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError, InvalidOutputSelectionError


# Trả về thời gian UTC có timezone để mọi phép so sánh lease và retention nhất quán.
def utc_now() -> datetime:
    """Tạo timestamp UTC cho domain mà không phụ thuộc clock của framework."""

    return datetime.now(UTC)


# Đại diện một batch idempotent gồm nhiều sản phẩm của cùng seller.
@dataclass(frozen=True)
class ImageOptimizationBatch:
    """Lưu fingerprint request để cùng key không thể bị tái sử dụng cho payload khác."""

    batch_id: UUID
    seller_owner_id: UUID
    idempotency_key: str
    request_hash: str
    created_at: datetime = field(default_factory=utc_now)

    # Tạo batch mới sau khi use case đã chuẩn hóa và xác minh request.
    @classmethod
    def create(cls, *, seller_owner_id: UUID, idempotency_key: str, request_hash: str) -> "ImageOptimizationBatch":
        """Tạo identity độc lập với idempotency key do client cung cấp."""

        normalized_key = idempotency_key.strip()
        if len(normalized_key) < 8 or len(request_hash) != 64:
            raise ValueError("Invalid image optimization batch identity")
        return cls(
            batch_id=uuid4(),
            seller_owner_id=seller_owner_id,
            idempotency_key=normalized_key,
            request_hash=request_hash,
        )


# Tham chiếu một output đã được Media Service lưu và gắn đúng ảnh nguồn.
@dataclass(frozen=True)
class GeneratedAsset:
    """Không chứa binary; public URL chỉ là dữ liệu tương thích tạm thời cho API hiện tại."""

    asset_id: UUID
    public_url: str | None
    mode: str
    output_id: UUID = field(default_factory=uuid4)
    source_asset_id: UUID | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


ALLOWED_TRANSITIONS: dict[ImageOptimizationStatus, frozenset[ImageOptimizationStatus]] = {
    ImageOptimizationStatus.PENDING: frozenset({ImageOptimizationStatus.PROCESSING, ImageOptimizationStatus.FAILED}),
    ImageOptimizationStatus.PROCESSING: frozenset({ImageOptimizationStatus.REVIEW_REQUIRED, ImageOptimizationStatus.FAILED}),
    ImageOptimizationStatus.REVIEW_REQUIRED: frozenset(
        {ImageOptimizationStatus.APPLIED, ImageOptimizationStatus.REJECTED, ImageOptimizationStatus.FAILED}
    ),
    # SUCCEEDED chỉ được giữ để đọc dữ liệu cũ; domain mới không tạo trạng thái này.
    ImageOptimizationStatus.SUCCEEDED: frozenset({ImageOptimizationStatus.APPLIED, ImageOptimizationStatus.REJECTED}),
    ImageOptimizationStatus.APPLIED: frozenset({ImageOptimizationStatus.ROLLED_BACK}),
    ImageOptimizationStatus.REJECTED: frozenset(),
    ImageOptimizationStatus.ROLLED_BACK: frozenset(),
    ImageOptimizationStatus.FAILED: frozenset({ImageOptimizationStatus.PENDING, ImageOptimizationStatus.PROCESSING}),
}


# Aggregate quản lý vòng đời tối ưu của đúng một sản phẩm trong batch.
@dataclass(frozen=True)
class ImageOptimizationJob:
    """Bảo vệ state transition, output selection, lease và retention của một job."""

    job_id: UUID
    seller_owner_id: UUID
    product_id: UUID
    source_asset_ids: tuple[UUID, ...]
    requested_modes: tuple[ImageOptimizationMode, ...]
    idempotency_key: str
    expected_product_updated_at: datetime | None
    batch_id: UUID | None = None
    request_hash: str | None = None
    background_preset: LifestyleBackgroundPreset | None = None
    background_description_ciphertext: str | None = None
    background_description_hash: str | None = None
    status: ImageOptimizationStatus = ImageOptimizationStatus.PENDING
    processing_stage: ImageOptimizationProcessingStage = ImageOptimizationProcessingStage.QUEUED
    generated_asset_ids: tuple[UUID, ...] = ()
    generated_assets: tuple[GeneratedAsset, ...] = ()
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    attempt: int = 0
    version: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    failure_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retention_expires_at: datetime | None = None

    # Tạo aggregate mới sau khi ownership và asset selection đã được application xác minh.
    @classmethod
    def create(
        cls,
        *,
        seller_owner_id: UUID,
        product_id: UUID,
        source_asset_ids: tuple[UUID, ...],
        requested_modes: tuple[ImageOptimizationMode, ...],
        idempotency_key: str,
        expected_product_updated_at: datetime | None,
        batch_id: UUID | None = None,
        request_hash: str | None = None,
        background_preset: LifestyleBackgroundPreset | None = None,
        background_description_ciphertext: str | None = None,
        background_description_hash: str | None = None,
    ) -> "ImageOptimizationJob":
        """Chặn source/mode rỗng và giữ identity client tách khỏi job ID nội bộ."""

        if not source_asset_ids:
            raise ValueError("At least one source asset is required")
        if not requested_modes:
            raise ValueError("At least one optimization mode is required")
        if len(idempotency_key.strip()) < 8:
            raise ValueError("Idempotency key is too short")
        return cls(
            job_id=uuid4(),
            seller_owner_id=seller_owner_id,
            product_id=product_id,
            source_asset_ids=source_asset_ids,
            requested_modes=requested_modes,
            idempotency_key=idempotency_key.strip(),
            expected_product_updated_at=expected_product_updated_at,
            batch_id=batch_id,
            request_hash=request_hash,
            background_preset=background_preset,
            background_description_ciphertext=background_description_ciphertext,
            background_description_hash=background_description_hash,
        )

    # Cập nhật stage để frontend theo dõi tiến trình nhưng không thay đổi trạng thái nghiệp vụ.
    def with_processing_stage(self, processing_stage: ImageOptimizationProcessingStage) -> "ImageOptimizationJob":
        """Trả aggregate mới, tăng version để persistence phát hiện ghi đè cũ."""

        return replace(self, processing_stage=processing_stage, version=self.version + 1)

    # Chuyển trạng thái theo state machine và ghi timestamp hoàn tất khi cần.
    def transition(self, next_status: ImageOptimizationStatus) -> "ImageOptimizationJob":
        """Từ chối mọi transition không được domain cho phép."""

        if next_status not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidJobTransitionError(f"Cannot move {self.status} to {next_status}")
        now = utc_now()
        started_at = now if next_status is ImageOptimizationStatus.PROCESSING else self.started_at
        completed_at = (
            now
            if next_status
            in {
                ImageOptimizationStatus.REVIEW_REQUIRED,
                ImageOptimizationStatus.SUCCEEDED,
                ImageOptimizationStatus.FAILED,
                ImageOptimizationStatus.REJECTED,
                ImageOptimizationStatus.APPLIED,
                ImageOptimizationStatus.ROLLED_BACK,
            }
            else self.completed_at
        )
        return replace(
            self,
            status=next_status,
            started_at=started_at,
            completed_at=completed_at,
            version=self.version + 1,
        )

    # Claim lease trước khi gọi provider để Kafka redelivery không tạo lời gọi trả phí trùng.
    def claim(self, *, worker_id: str, lease_seconds: int, now: datetime | None = None) -> "ImageOptimizationJob":
        """Chuyển job sang PROCESSING, tăng attempt và đặt thời hạn lease rõ ràng."""

        current_time = now or utc_now()
        if self.status is ImageOptimizationStatus.PROCESSING and self.lease_expires_at and self.lease_expires_at > current_time:
            raise InvalidJobTransitionError("Image optimization job is already leased")
        claimable_statuses = {
            ImageOptimizationStatus.PENDING,
            ImageOptimizationStatus.FAILED,
            ImageOptimizationStatus.PROCESSING,
        }
        if self.status not in claimable_statuses:
            raise InvalidJobTransitionError("Image optimization job cannot be claimed")
        return replace(
            self,
            status=ImageOptimizationStatus.PROCESSING,
            processing_stage=ImageOptimizationProcessingStage.FETCHING_SOURCE,
            attempt=self.attempt + 1,
            version=self.version + 1,
            lease_owner=worker_id,
            lease_expires_at=current_time + timedelta(seconds=max(1, lease_seconds)),
            started_at=self.started_at or current_time,
            completed_at=None,
            failure_code=None,
        )

    # Gắn output đã upload và thời hạn review; asset ID được suy ra từ một nguồn duy nhất.
    def with_outputs(
        self,
        assets: tuple[GeneratedAsset, ...],
        *,
        provider: str,
        model: str | None,
        prompt_version: str | None,
        retention_days: int,
    ) -> "ImageOptimizationJob":
        """Chặn output thiếu source mapping để không thể apply nhầm ảnh."""

        if not assets or any(asset.source_asset_id is None for asset in assets):
            raise ValueError("Every generated output must reference its source asset")
        return replace(
            self,
            generated_asset_ids=tuple(asset.asset_id for asset in assets),
            generated_assets=assets,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            retention_expires_at=utc_now() + timedelta(days=max(1, retention_days)),
            version=self.version + 1,
        )

    # Lọc đúng output seller chọn và không tin URL/mode do browser gửi lên.
    def select_outputs(self, asset_ids: tuple[UUID, ...]) -> tuple[GeneratedAsset, ...]:
        """Trả toàn bộ output khi danh sách rỗng để giữ tương thích API cũ."""

        if not asset_ids:
            return self.generated_assets
        if len(set(asset_ids)) != len(asset_ids):
            raise InvalidOutputSelectionError("Duplicate output assets are not allowed")
        output_by_asset = {asset.asset_id: asset for asset in self.generated_assets}
        try:
            return tuple(output_by_asset[asset_id] for asset_id in asset_ids)
        except KeyError as error:
            raise InvalidOutputSelectionError("Selected output does not belong to this job") from error

    # Gắn failure code đã redact và giải phóng lease để job có thể được xử lý lại có kiểm soát.
    def with_failure(self, failure_code: str) -> "ImageOptimizationJob":
        """Không lưu exception message, prompt hoặc provider response thô."""

        return replace(
            self,
            failure_code=failure_code[:80],
            lease_owner=None,
            lease_expires_at=None,
            version=self.version + 1,
        )

    # Giải phóng lease sau trạng thái terminal để worker khác không hiểu nhầm job còn chạy.
    def release_lease(self) -> "ImageOptimizationJob":
        """Xóa metadata lease nhưng giữ nguyên trạng thái và output."""

        return replace(self, lease_owner=None, lease_expires_at=None, version=self.version + 1)
