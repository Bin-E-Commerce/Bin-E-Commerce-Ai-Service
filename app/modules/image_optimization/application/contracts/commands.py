"""Command bất biến đi từ presentation vào các use case tối ưu ảnh.

Command không chứa FastAPI/Pydantic và cung cấp fingerprint ổn định phục vụ
idempotency mà không phải lưu payload request thô.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.modules.image_optimization.domain.enums import ImageOptimizationMode, LifestyleBackgroundPreset


@dataclass(frozen=True)
class CreateOptimizationJobsCommand:
    """Du lieu da parse de tao batch job theo seller."""

    seller_owner_id: UUID
    product_ids: tuple[UUID, ...]
    source_asset_policy: str
    modes: tuple[ImageOptimizationMode, ...]
    idempotency_key: str
    expected_product_updated_at: datetime | None
    source_asset_ids: tuple[UUID, ...] = ()
    background_preset: LifestyleBackgroundPreset | None = None
    background_description: str | None = None
    force_regenerate: bool = False
    permissions: frozenset[str] = frozenset()
    seller_email: str = ""

    # Hash mọi trường ảnh hưởng kết quả để cùng idempotency key không thể đại diện hai payload khác nhau.
    def request_hash(self) -> str:
        """Tạo SHA-256 deterministic, không đưa seller text hoặc asset ID thô vào log."""

        payload = {
            "sellerOwnerId": str(self.seller_owner_id),
            "productIds": sorted(str(value) for value in self.product_ids),
            "sourceAssetPolicy": self.source_asset_policy,
            "sourceAssetIds": sorted(str(value) for value in self.source_asset_ids),
            "modes": sorted(value.value for value in self.modes),
            "backgroundPreset": self.background_preset.value if self.background_preset else None,
            "backgroundDescriptionHash": (
                hashlib.sha256(self.background_description.encode("utf-8")).hexdigest() if self.background_description else None
            ),
            "forceRegenerate": self.force_regenerate,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Command apply chỉ mang dữ liệu seller đã xác nhận và permission đã được Gateway xác thực.
@dataclass(frozen=True)
class ApplyOptimizationOutputsCommand:
    """Không nhận URL từ browser; asset IDs sẽ được đối chiếu với output của job."""

    job_id: UUID
    seller_owner_id: UUID
    expected_product_updated_at: datetime
    selected_asset_ids: tuple[UUID, ...]
    permissions: frozenset[str]
    seller_email: str = ""
