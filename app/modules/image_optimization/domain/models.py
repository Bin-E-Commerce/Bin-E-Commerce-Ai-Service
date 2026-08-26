"""Aggregate va value object cho job toi uu anh, khong luu binary hay signed URL."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.modules.image_optimization.domain.enums import (
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
    LifestyleBackgroundPreset,
)
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError


@dataclass(frozen=True)
class GeneratedAsset:
    """Tham chieu output da duoc Media Service luu, khong luu binary trong domain."""

    asset_id: UUID
    public_url: str | None
    mode: str


ALLOWED_TRANSITIONS: dict[ImageOptimizationStatus, frozenset[ImageOptimizationStatus]] = {
    ImageOptimizationStatus.PENDING: frozenset({ImageOptimizationStatus.PROCESSING, ImageOptimizationStatus.FAILED}),
    ImageOptimizationStatus.PROCESSING: frozenset({ImageOptimizationStatus.REVIEW_REQUIRED, ImageOptimizationStatus.FAILED}),
    ImageOptimizationStatus.REVIEW_REQUIRED: frozenset(
        {
            ImageOptimizationStatus.SUCCEEDED,
            ImageOptimizationStatus.APPLIED,
            ImageOptimizationStatus.REJECTED,
            ImageOptimizationStatus.FAILED,
        }
    ),
    ImageOptimizationStatus.SUCCEEDED: frozenset({ImageOptimizationStatus.APPLIED, ImageOptimizationStatus.REJECTED}),
    ImageOptimizationStatus.APPLIED: frozenset({ImageOptimizationStatus.ROLLED_BACK}),
    ImageOptimizationStatus.REJECTED: frozenset(),
    ImageOptimizationStatus.ROLLED_BACK: frozenset(),
    ImageOptimizationStatus.FAILED: frozenset({ImageOptimizationStatus.PENDING}),
}


def utc_now() -> datetime:
    """Tra ve thoi gian UTC co timezone de tranh so sanh datetime khac lo."""

    return datetime.now(UTC)


@dataclass(frozen=True)
class ImageOptimizationJob:
    """Aggregate luu metadata cua mot lan toi uu mot san pham."""

    job_id: UUID
    seller_owner_id: UUID
    product_id: UUID
    source_asset_ids: tuple[UUID, ...]
    requested_modes: tuple[ImageOptimizationMode, ...]
    idempotency_key: str
    expected_product_updated_at: datetime | None
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
    failure_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retention_expires_at: datetime | None = None

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
        background_preset: LifestyleBackgroundPreset | None = None,
        background_description_ciphertext: str | None = None,
        background_description_hash: str | None = None,
    ) -> "ImageOptimizationJob":
        """Tao aggregate moi va chan input rong truoc khi ghi persistence."""

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
            background_preset=background_preset,
            background_description_ciphertext=background_description_ciphertext,
            background_description_hash=background_description_hash,
        )

    # Cập nhật bước xử lý riêng với trạng thái nghiệp vụ để UI phản hồi ngay cả khi job vẫn PROCESSING.
    def with_processing_stage(self, processing_stage: ImageOptimizationProcessingStage) -> "ImageOptimizationJob":
        """Trả aggregate mới chứa chặng xử lý an toàn để seller theo dõi."""

        return self.__class__(**{**self.__dict__, "processing_stage": processing_stage})

    def transition(self, next_status: ImageOptimizationStatus) -> "ImageOptimizationJob":
        """Chuyen state theo state machine, tranh apply/retry sai thu tu."""

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
        return self.__class__(
            **{
                **self.__dict__,
                "status": next_status,
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )

    def with_outputs(
        self,
        asset_ids: tuple[UUID, ...],
        provider: str,
        model: str | None,
        prompt_version: str | None,
        assets: tuple[GeneratedAsset, ...] = (),
    ) -> "ImageOptimizationJob":
        """Gan output asset IDs va provider metadata ma khong luu raw output."""

        return self.__class__(
            **{
                **self.__dict__,
                "generated_asset_ids": asset_ids,
                "generated_assets": assets,
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
            }
        )

    def with_failure(self, failure_code: str) -> "ImageOptimizationJob":
        """Gan ma loi cong khai da redact de retry va quan sat an toan."""

        return self.__class__(**{**self.__dict__, "failure_code": failure_code})
