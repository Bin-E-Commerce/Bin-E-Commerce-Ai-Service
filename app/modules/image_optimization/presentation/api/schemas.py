"""Pydantic request/response schemas, chỉ validate shape tại HTTP boundary."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.image_optimization.domain.enums import (
    ImageGenerationProfile,
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
    LifestyleBackgroundPreset,
)


class SourceAssetPolicy(StrEnum):
    """Chinh sach chon anh dau vao de worker resolve qua Product/Media Service."""

    COVER_IMAGE = "COVER_IMAGE"
    SELECTED_ASSETS = "SELECTED_ASSETS"


# Đóng gói lựa chọn bối cảnh để request không truyền prompt hệ thống hoặc điều khiển model tùy ý từ browser.
class LifestyleBackgroundRequestSchema(BaseModel):
    """Dữ liệu bối cảnh do seller kiểm soát trong giới hạn nhỏ, an toàn cho ảnh lifestyle."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    preset: LifestyleBackgroundPreset | None = None
    description: str | None = Field(default=None, max_length=400)

    # Chuẩn hóa chuỗi để hash/cache không khác nhau chỉ vì khoảng trắng đầu cuối.
    @model_validator(mode="after")
    def normalize_description(self) -> "LifestyleBackgroundRequestSchema":
        """Cắt khoảng trắng và bỏ chuỗi rỗng để application chỉ xử lý intent thực sự của seller."""

        if self.description is not None:
            normalized = " ".join(self.description.split())
            if normalized and len(normalized) < 10:
                raise ValueError("Background description must be at least 10 characters or empty")
            self.description = normalized or None
        return self


class CreateImageOptimizationRequest(BaseModel):
    """Payload tao batch job khong cho phep frontend gui binary hoac URL."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    product_ids: list[UUID] = Field(min_length=1, max_length=1, alias="productIds")
    source_asset_ids: list[UUID] | None = Field(default=None, min_length=1, max_length=1, alias="sourceAssetIds")
    source_asset_policy: SourceAssetPolicy = Field(default=SourceAssetPolicy.COVER_IMAGE, alias="sourceAssetPolicy")
    modes: list[ImageOptimizationMode] = Field(min_length=1, max_length=2)
    background: LifestyleBackgroundRequestSchema | None = None
    force_regenerate: bool = Field(default=False, alias="forceRegenerate")

    # Kiểm tra quan hệ giữa mode, batch và background ở HTTP boundary trước khi tạo job hoặc tốn quota.
    @model_validator(mode="after")
    def validate_background_intent(self) -> "CreateImageOptimizationRequest":
        """Chỉ cho phép background ở lifestyle và chỉ một sản phẩm khi seller viết mô tả riêng."""

        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Optimization modes must be unique")
        if self.source_asset_policy is SourceAssetPolicy.SELECTED_ASSETS and not self.source_asset_ids:
            raise ValueError("Selected assets are required for SELECTED_ASSETS policy")
        if self.source_asset_ids and self.source_asset_policy is not SourceAssetPolicy.SELECTED_ASSETS:
            raise ValueError("sourceAssetIds require SELECTED_ASSETS policy")
        if self.source_asset_ids and len(self.product_ids) != 1:
            raise ValueError("Selected source assets require exactly one product")
        if self.background and ImageOptimizationMode.LIFESTYLE_BACKGROUND not in self.modes:
            raise ValueError("Background options are only available for lifestyle optimization")
        if self.background and self.background.description and len(self.product_ids) != 1:
            raise ValueError("Custom lifestyle background requires exactly one product")
        return self


class OptimizationJobResponse(BaseModel):
    """Thong tin job seller can de theo doi va mo preview."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: UUID = Field(alias="jobId")
    product_id: UUID = Field(alias="productId")
    status: ImageOptimizationStatus
    processing_stage: ImageOptimizationProcessingStage = Field(alias="processingStage")
    generation_profile: ImageGenerationProfile = Field(alias="generationProfile")
    background_preset: LifestyleBackgroundPreset | None = Field(default=None, alias="backgroundPreset")
    generated_asset_ids: list[UUID] = Field(alias="generatedAssetIds")
    generated_assets: list[dict[str, str | None]] = Field(default_factory=list, alias="generatedAssets")
    created_at: datetime = Field(alias="createdAt")
    expected_product_updated_at: datetime | None = Field(default=None, alias="expectedProductUpdatedAt")
    failure_code: str | None = Field(default=None, alias="failureCode")


# Mô tả output seller chọn bằng asset ID; URL do browser gửi không được tin cậy.
class ApplyImageSelection(BaseModel):
    """Giữ tương thích với payload hiện tại nhưng chỉ assetId được dùng để đối chiếu."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    asset_id: UUID = Field(alias="assetId")


class CreateImageOptimizationResponse(BaseModel):
    """Response 202 cho batch tao job."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(alias="batchId")
    jobs: list[OptimizationJobResponse]


class ImageOptimizationUsageResponse(BaseModel):
    """Quota snapshot for the current seller and current sliding window."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool
    limit: int | None
    used: int
    remaining: int | None


class ImageOptimizationOverviewResponse(BaseModel):
    """Metric dashboard lay tu persistence, khong dung so lieu placeholder."""

    model_config = ConfigDict(populate_by_name=True)

    optimized_products: int | None = Field(default=None, alias="optimizedProducts")
    total_views: int | None = Field(default=None, alias="totalViews")
    total_sold: int | None = Field(default=None, alias="totalSold")
    pending_jobs: int = Field(alias="pendingJobs")
    failed_jobs: int = Field(alias="failedJobs")
    ai_usage: ImageOptimizationUsageResponse = Field(alias="aiUsage")


class ImageOptimizationImpactMetricResponse(BaseModel):
    """Metric trung bình baseline so với ngày hậu tối ưu gần nhất."""

    model_config = ConfigDict(populate_by_name=True)

    before: float
    after: float
    delta: float
    change_percent: float | None = Field(default=None, alias="changePercent")


class ImageOptimizationImpactDailyResponse(BaseModel):
    """Một ngày hậu tối ưu, dùng để seller thấy trend mà không phải chờ đủ window."""

    model_config = ConfigDict(populate_by_name=True)

    date: str
    views: int
    sales: int
    has_data: bool = Field(alias="hasData")
    views_change_percent: float | None = Field(default=None, alias="viewsChangePercent")
    sales_change_percent: float | None = Field(default=None, alias="salesChangePercent")


class ImageOptimizationImpactOverviewResponse(BaseModel):
    """KPI baseline/ngày gần nhất toàn seller, không khẳng định quan hệ nhân quả."""

    model_config = ConfigDict(populate_by_name=True)

    tracked_products: int = Field(alias="trackedProducts")
    ready_products: int = Field(alias="readyProducts")
    collecting_products: int = Field(alias="collectingProducts")
    no_baseline_products: int = Field(alias="noBaselineProducts")
    views: ImageOptimizationImpactMetricResponse
    sales: ImageOptimizationImpactMetricResponse
    status: str
    window_days: int | None = Field(default=None, alias="windowDays")
    baseline_days: int | None = Field(default=None, alias="baselineDays")


class ImageOptimizationProductImpactResponse(BaseModel):
    """Baseline và trend daily của một sản phẩm theo job apply mới nhất."""

    model_config = ConfigDict(populate_by_name=True)

    product_id: UUID = Field(alias="productId")
    job_id: UUID | None = Field(default=None, alias="jobId")
    applied_at: datetime | None = Field(default=None, alias="appliedAt")
    status: str
    elapsed_seconds: int = Field(alias="elapsedSeconds")
    days_collected: int = Field(alias="daysCollected")
    window_days: int = Field(alias="windowDays")
    baseline_days: int = Field(alias="baselineDays")
    views: ImageOptimizationImpactMetricResponse | None = None
    sales: ImageOptimizationImpactMetricResponse | None = None
    daily: list[ImageOptimizationImpactDailyResponse] = Field(default_factory=list)


class ImageOptimizationProductImpactsResponse(BaseModel):
    """Danh sách impact đã lọc theo seller và product IDs hợp lệ."""

    model_config = ConfigDict(populate_by_name=True)

    items: list[ImageOptimizationProductImpactResponse]


class ApplyImageOptimizationRequest(BaseModel):
    """Thong tin concurrency va output asset seller da xac nhan."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    expected_product_updated_at: datetime = Field(alias="expectedProductUpdatedAt")
    images: list[ApplyImageSelection] = Field(default_factory=list, max_length=9)
