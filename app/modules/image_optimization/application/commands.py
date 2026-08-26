"""Command immutable tu presentation vao application layer."""

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
