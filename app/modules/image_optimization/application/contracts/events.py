"""Versioned event contract cho yêu cầu tối ưu ảnh.

Contract chỉ chứa metadata cần cho worker, không chứa binary, URL, JWT, email,
permission, prompt hoặc dữ liệu seller nhập.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from app.modules.image_optimization.domain.enums import ImageGenerationProfile
from app.modules.image_optimization.domain.models import ImageOptimizationJob, utc_now


# Event immutable dùng chung cho memory publisher, outbox và Kafka adapter.
@dataclass(frozen=True)
class ImageOptimizationRequestedEvent:
    """Một event ID duy nhất được giữ nguyên từ database tới Kafka consumer."""

    event_id: UUID
    job_id: UUID
    product_id: UUID
    seller_owner_id: UUID
    source_asset_ids: tuple[UUID, ...]
    modes: tuple[str, ...]
    occurred_at: datetime
    event_type: str = "ai.image-optimization.requested"
    schema_version: int = 1
    generation_profile: ImageGenerationProfile = ImageGenerationProfile.PREVIEW

    # Tạo event từ job sau khi job đã có identity persistence.
    @classmethod
    def from_job(cls, job: ImageOptimizationJob, *, event_id: UUID | None = None) -> "ImageOptimizationRequestedEvent":
        """Không copy background description hoặc provider config vào broker."""

        return cls(
            event_id=event_id or uuid4(),
            job_id=job.job_id,
            product_id=job.product_id,
            seller_owner_id=job.seller_owner_id,
            source_asset_ids=job.source_asset_ids[:1]
            if job.generation_profile is ImageGenerationProfile.PREVIEW
            else job.source_asset_ids,
            modes=tuple(mode.value for mode in job.requested_modes),
            occurred_at=utc_now(),
            generation_profile=job.generation_profile,
        )

    # Chuyển event sang wire payload camelCase ổn định cho Kafka.
    def to_payload(self) -> dict[str, object]:
        """Giữ đúng một encoder để các publisher không tạo contract khác nhau."""

        return {
            "eventId": str(self.event_id),
            "eventType": self.event_type,
            "schemaVersion": self.schema_version,
            "jobId": str(self.job_id),
            "productId": str(self.product_id),
            "sellerOwnerId": str(self.seller_owner_id),
            "sourceAssetIds": [str(value) for value in self.source_asset_ids],
            "modes": list(self.modes),
            "generationProfile": self.generation_profile.value,
            "occurredAt": self.occurred_at.isoformat(),
        }

    # Parse wire payload bằng một schema dùng chung để consumer không tự đọc riêng mỗi trường.
    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ImageOptimizationRequestedEvent":
        """Từ chối event type/version sai trước khi worker mở transaction hoặc gọi provider."""

        if payload.get("eventType") != "ai.image-optimization.requested" or payload.get("schemaVersion") != 1:
            raise ValueError("Unsupported image optimization event schema")
        source_values = payload.get("sourceAssetIds")
        mode_values = payload.get("modes")
        if not isinstance(source_values, list) or not isinstance(mode_values, list):
            raise ValueError("Invalid image optimization event collections")
        return cls(
            event_id=UUID(str(payload["eventId"])),
            job_id=UUID(str(payload["jobId"])),
            product_id=UUID(str(payload["productId"])),
            seller_owner_id=UUID(str(payload["sellerOwnerId"])),
            source_asset_ids=tuple(UUID(str(value)) for value in source_values),
            modes=tuple(str(value) for value in mode_values),
            occurred_at=datetime.fromisoformat(str(payload["occurredAt"])),
            generation_profile=ImageGenerationProfile(str(payload.get("generationProfile", "PREVIEW"))),
        )
