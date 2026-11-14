"""HTTP contract cho query embedding nội bộ, không bao gồm nội dung catalog."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SearchQueryEmbeddingRequest(BaseModel):
    """Query tìm kiếm bị chặn độ dài trước khi gọi embedding provider."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=120)]


class SearchQueryEmbeddingResponse(BaseModel):
    """Vector kèm model/dimension để Recommendation Service kiểm tra tương thích."""

    model_config = ConfigDict(populate_by_name=True)

    model_version: str = Field(alias="modelVersion")
    dimensions: int
    vector: list[float]
