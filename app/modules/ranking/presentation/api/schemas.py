"""Pydantic schemas cho batch ranking, giới hạn payload tại HTTP boundary."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class RankingItemRequest(BaseModel):
    """Candidate ID và feature vector đã được Recommendation normalize."""

    model_config = ConfigDict(extra="forbid")

    item_id: Annotated[str, Field(min_length=1, max_length=128, alias="itemId")]
    features: Annotated[list[float], Field(min_length=1, max_length=64)]


class RankingPredictRequest(BaseModel):
    """Batch request nội bộ, không nhận prompt/catalog text hoặc user data."""

    model_config = ConfigDict(extra="forbid")

    request_id: Annotated[str, Field(min_length=1, max_length=128, alias="requestId")]
    items: Annotated[list[RankingItemRequest], Field(min_length=1, max_length=300)]


class RankingPredictionResponse(BaseModel):
    """Response score theo đúng thứ tự input và model version đang phục vụ."""

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    model_version: str = Field(alias="modelVersion")
    predictions: list[dict[str, str | float]]
