"""Schema phản hồi embedding ảnh; request bytes được kiểm tra trực tiếp ở router."""

from pydantic import BaseModel, ConfigDict, Field


class SearchImageEmbeddingResponse(BaseModel):
    """Model version và dimension cho Recommendation xác nhận đúng vector space."""

    model_config = ConfigDict(populate_by_name=True)

    model_version: str = Field(alias="modelVersion")
    dimensions: int
    preprocess_version: str = Field(alias="preprocessVersion")
    vector: list[float]
    detail_vectors: tuple[list[float], list[float]] | None = Field(default=None, alias="detailVectors")
    rotation_vectors: tuple[list[float], list[float]] | None = Field(default=None, alias="rotationVectors")
    strong_match_required: bool | None = Field(default=None, alias="strongMatchRequired")


class SearchImageEmbeddingStatusResponse(BaseModel):
    """Readiness nội bộ không tiết lộ artifact path hay nội dung ảnh."""

    model_config = ConfigDict(populate_by_name=True)

    ready: bool
    model_version: str | None = Field(alias="modelVersion")
    dimensions: int | None
    preprocess_version: str = Field(alias="preprocessVersion")
