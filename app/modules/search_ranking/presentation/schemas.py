"""Request/response contract giới hạn văn bản và số candidate của search reranker."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SearchRerankItemRequest(BaseModel):
    """Một candidate catalog tối thiểu; không nhận user profile hay dữ liệu thanh toán."""

    model_config = ConfigDict(extra="forbid")

    item_id: Annotated[str, Field(min_length=1, max_length=128, alias="itemId")]
    title: Annotated[str, Field(min_length=1, max_length=500)]
    brand: Annotated[str | None, Field(max_length=180)] = None
    category: Annotated[str | None, Field(max_length=300)] = None
    sku: Annotated[str | None, Field(max_length=64)] = None
    description: Annotated[str | None, Field(max_length=2400)] = None


class SearchRerankRequest(BaseModel):
    """Batch query-product pairs; query và candidate text không được ghi log."""

    model_config = ConfigDict(extra="forbid")

    request_id: Annotated[str, Field(min_length=1, max_length=128, alias="requestId")]
    query: Annotated[str, Field(min_length=1, max_length=120)]
    items: Annotated[list[SearchRerankItemRequest], Field(min_length=1, max_length=100)]


class SearchRerankPredictionResponse(BaseModel):
    """Raw model score giữ nguyên item ID để caller ổn định tie-break."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str = Field(alias="itemId")
    score: float


class SearchRerankResponse(BaseModel):
    """Phiên bản artifact kèm kết quả theo thứ tự request."""

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    model_version: str = Field(alias="modelVersion")
    predictions: list[SearchRerankPredictionResponse]


class SearchRerankerStatusResponse(BaseModel):
    """Readiness tối thiểu, không tiết lộ đường dẫn file hoặc cấu hình host."""

    model_config = ConfigDict(populate_by_name=True)

    ready: bool
    model_version: str | None = Field(alias="modelVersion")
    schema_version: str = Field(alias="schemaVersion")
