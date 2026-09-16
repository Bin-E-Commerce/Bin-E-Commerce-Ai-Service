"""Router internal cho Recommendation gọi batch ML score qua service token."""

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, ConfigurationError
from app.modules.ranking.application.service import RankingPredictionService
from app.modules.ranking.presentation.api.schemas import (
    RankingPredictionResponse,
    RankingPredictRequest,
    RankingStatusResponse,
)

router = APIRouter(prefix="/api/v1/ranking", tags=["ranking"])


def require_internal_token(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Header(alias="x-internal-service-token")] = None,
) -> None:
    """Chỉ cho service token đã cấu hình gọi endpoint, không dùng JWT/user permission của browser."""

    expected = settings.internal_service_token.get_secret_value() if settings.internal_service_token else ""
    if not expected:
        raise ConfigurationError()
    if not token or not hmac.compare_digest(token, expected):
        raise AuthenticationError()


def get_ranking_prediction_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RankingPredictionService:
    """Lấy model registry từ app state để không load artifact trên mỗi request."""

    model = getattr(request.app.state, "ranking_model", None)
    if model is None:
        raise ConfigurationError()
    return RankingPredictionService(
        model,
        settings.ranking_max_items,
        settings.ranking_max_features,
        settings.ranking_expected_features,
    )


@router.post("/predict", response_model=RankingPredictionResponse, response_model_by_alias=True)
async def predict(
    payload: RankingPredictRequest,
    _: Annotated[None, Depends(require_internal_token)],
    service: Annotated[RankingPredictionService, Depends(get_ranking_prediction_service)],
) -> RankingPredictionResponse:
    """Validate batch, gọi use case và map output an toàn ra HTTP response."""

    result = service.predict(
        payload.request_id,
        [(item.item_id, item.features) for item in payload.items],
    )
    return RankingPredictionResponse(
        request_id=result.request_id,
        model_version=result.model_version,
        predictions=[{"itemId": item.item_id, "score": item.score} for item in result.predictions],
    )


@router.get("/status", response_model=RankingStatusResponse, response_model_by_alias=True)
async def status(
    _: Annotated[None, Depends(require_internal_token)],
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RankingStatusResponse:
    """Trả model version và trạng thái fallback để Recommendation/Admin biết AI có thực sự sẵn sàng."""

    model = getattr(request.app.state, "ranking_model", None)
    if model is None:
        raise ConfigurationError()
    model_version = str(getattr(model, "model_version", "ranking-fallback-v1"))
    fallback = model_version.startswith("ranking-fallback")
    return RankingStatusResponse(
        ready=not fallback,
        fallback=fallback,
        model_version=model_version,
        feature_count=settings.ranking_expected_features,
    )
