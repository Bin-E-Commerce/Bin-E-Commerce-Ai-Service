"""Pydantic schemas cho batch ranking, giới hạn payload và khóa phiên bản feature tại HTTP boundary."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.modules.ranking.domain.models import RANKING_FEATURE_SCHEMA_VERSION


class RankingItemRequest(BaseModel):
    """Candidate ID và feature vector đã được Recommendation normalize."""

    model_config = ConfigDict(extra="forbid")

    item_id: Annotated[str, Field(min_length=1, max_length=128, alias="itemId")]
    features: Annotated[list[float], Field(min_length=1, max_length=64)]


# Nhận batch chỉ khi phiên bản feature khớp artifact hiện hành; từ chối trước bước suy luận.
class RankingPredictRequest(BaseModel):
    """Batch request nội bộ, không nhận prompt/catalog text hoặc user data."""

    model_config = ConfigDict(extra="forbid")

    request_id: Annotated[str, Field(min_length=1, max_length=128, alias="requestId")]
    feature_schema_version: str = Field(alias="featureSchemaVersion")
    items: Annotated[list[RankingItemRequest], Field(min_length=1, max_length=300)]

    # Chặn deploy lệch phiên bản: cùng chín cột nhưng semantics sessionContext v2/v3 khác nhau.
    # Dùng constant chung với model registry để HTTP contract không tự lệch khi nâng schema tiếp.
    @field_validator("feature_schema_version")
    @classmethod
    def require_current_feature_schema(cls, version: str) -> str:
        if version != RANKING_FEATURE_SCHEMA_VERSION:
            raise ValueError("RANKING_FEATURE_SCHEMA_VERSION_MISMATCH")
        return version


class RankingPredictionResponse(BaseModel):
    """Response raw LambdaRank relevance score theo đúng thứ tự input."""

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    model_version: str = Field(alias="modelVersion")
    predictions: list[dict[str, str | float]]


class RankingStatusResponse(BaseModel):
    """Trạng thái model ranking nội bộ, không tiết lộ đường dẫn artifact hoặc secret."""

    model_config = ConfigDict(populate_by_name=True)

    ready: bool
    model_version: str | None = Field(alias="modelVersion")
    feature_count: int = Field(alias="featureCount")
