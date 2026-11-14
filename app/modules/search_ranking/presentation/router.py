"""Endpoint nội bộ batch scoring dành riêng cho Product Service."""

import hmac
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidInputError,
    ProviderUnavailableError,
    SearchRerankerUnavailableError,
)
from app.modules.search_ranking.application.query_embedding_service import SearchQueryEmbeddingService
from app.modules.search_ranking.application.service import SearchRerankingService
from app.modules.search_ranking.domain.models import SEARCH_RERANKER_SCHEMA_VERSION
from app.modules.search_ranking.presentation.query_embedding_schemas import (
    SearchQueryEmbeddingRequest,
    SearchQueryEmbeddingResponse,
)
from app.modules.search_ranking.presentation.schemas import (
    SearchRerankerStatusResponse,
    SearchRerankItemRequest,
    SearchRerankPredictionResponse,
    SearchRerankRequest,
    SearchRerankResponse,
)

router = APIRouter(prefix="/api/v1/search-ranking", tags=["search-ranking"])


# Chỉ chấp nhận service token so sánh constant-time; endpoint không dành cho browser hoặc seller.
def require_internal_token(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Header(alias="x-internal-service-token")] = None,
) -> None:
    """Từ chối khi token thiếu cấu hình hoặc không khớp credential nội bộ."""

    expected = settings.internal_service_token.get_secret_value() if settings.internal_service_token else ""
    if not expected:
        raise ConfigurationError()
    if not token or not hmac.compare_digest(token, expected):
        raise AuthenticationError()


# Lấy model immutable từ app state để không load artifact trong từng request.
def get_search_reranking_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchRerankingService:
    """Tạo use case nhẹ quanh model dùng chung; model vắng mặt trả 503 rõ ràng."""

    model = getattr(request.app.state, "search_reranker", None)
    if model is None:
        raise SearchRerankerUnavailableError()
    return SearchRerankingService(model, settings.search_reranker_max_items, settings.search_reranker_max_product_chars)


# Gắn provider và Redis client từ lifespan; thiếu OpenAI key chỉ tắt semantic query expansion.
def get_query_embedding_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchQueryEmbeddingService:
    """Tạo use case cache/vector quanh provider dùng chung của application."""

    provider = getattr(request.app.state, "search_query_embedding_provider", None)
    if provider is None:
        raise ProviderUnavailableError()
    return SearchQueryEmbeddingService(
        provider,
        getattr(request.app.state, "redis_client", None),
        settings.search_query_embedding_cache_ttl_seconds,
    )


@router.post("/rerank", response_model=SearchRerankResponse, response_model_by_alias=True)
async def rerank(
    payload: SearchRerankRequest,
    _: Annotated[None, Depends(require_internal_token)],
    service: Annotated[SearchRerankingService, Depends(get_search_reranking_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchRerankResponse:
    """Chấm điểm một nhóm ứng viên; chạy inference ngoài event loop để giữ HTTP responsive."""

    request_id = payload.request_id.strip() or str(uuid4())
    result = await run_in_threadpool(
        service.rerank,
        request_id,
        payload.query,
        [_candidate_text(item, settings.search_reranker_max_product_chars) for item in payload.items],
    )
    return SearchRerankResponse(
        request_id=result.request_id,
        model_version=result.model_version,
        predictions=[SearchRerankPredictionResponse(item_id=item.item_id, score=item.score) for item in result.predictions],
    )


# Nối các trường public trong cùng một thứ tự và chặn tổng ký tự trước khi tokenizer xử lý.
def _candidate_text(item: SearchRerankItemRequest, limit: int) -> tuple[str, str]:
    """Tạo đúng cặp candidate ID/text từ schema đã validate ở HTTP boundary."""

    values = (
        item.title,
        item.brand,
        item.category,
        item.sku,
        item.description,
    )
    text = " ".join(value.strip() for value in values if value and value.strip())
    return item.item_id, text[:limit]


@router.post(
    "/query-embedding",
    response_model=SearchQueryEmbeddingResponse,
    response_model_by_alias=True,
)
async def embed_query(
    payload: SearchQueryEmbeddingRequest,
    _: Annotated[None, Depends(require_internal_token)],
    service: Annotated[SearchQueryEmbeddingService, Depends(get_query_embedding_service)],
) -> SearchQueryEmbeddingResponse:
    """Tạo vector cùng embedding catalog; raw query chỉ tồn tại trong request và cache key băm."""

    query = payload.query.strip()
    if not query:
        raise InvalidInputError()
    result = await service.embed(query)
    return SearchQueryEmbeddingResponse(
        model_version=result.model_version,
        dimensions=result.dimensions,
        vector=list(result.vector),
    )


@router.get("/status", response_model=SearchRerankerStatusResponse, response_model_by_alias=True)
async def status(
    _: Annotated[None, Depends(require_internal_token)],
    request: Request,
) -> SearchRerankerStatusResponse:
    """Cho caller biết artifact có sẵn và contract nào đang được phục vụ."""

    model = getattr(request.app.state, "search_reranker", None)
    return SearchRerankerStatusResponse(
        ready=model is not None,
        model_version=(str(model.model_version) if model is not None else None),
        schema_version=SEARCH_RERANKER_SCHEMA_VERSION,
    )
