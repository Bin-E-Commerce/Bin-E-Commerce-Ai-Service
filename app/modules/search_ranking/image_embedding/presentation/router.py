"""HTTP nội bộ nhận raw image bytes và trả embedding PE-Core đã validate."""

import hmac
import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Header, Query, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, ConfigurationError, InvalidInputError, ProviderUnavailableError
from app.modules.search_ranking.image_embedding.application.image_preprocessing import (
    IMAGE_PREPROCESS_VERSION,
    ImageFraming,
    InvalidImageData,
)
from app.modules.search_ranking.image_embedding.application.service import QueryImageEmbedding, SearchImageEmbeddingService
from app.modules.search_ranking.image_embedding.presentation.schemas import (
    SearchImageEmbeddingResponse,
    SearchImageEmbeddingStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/search-ranking/image-embedding", tags=["search-ranking-image"])
_SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


# So sánh token nội bộ constant-time; endpoint raw ảnh không được gọi trực tiếp từ browser.
def require_internal_image_token(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Header(alias="x-internal-service-token")] = None,
) -> None:
    expected = settings.internal_service_token.get_secret_value() if settings.internal_service_token else ""
    if not expected:
        raise ConfigurationError()
    if not token or not hmac.compare_digest(token, expected):
        raise AuthenticationError()


# Stream request theo chunk và dừng ngay khi vượt byte cap, kể cả request không khai báo Content-Length.
async def _read_bounded_body(request: Request, max_bytes: int) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length)
        except ValueError as error:
            raise InvalidInputError() from error
        if declared_length < 1 or declared_length > max_bytes:
            raise InvalidInputError()
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise InvalidInputError()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise InvalidInputError()
        chunks.append(chunk)
    if total == 0:
        raise InvalidInputError()
    return b"".join(chunks)


# Trả 503 khi artifact vắng mặt để Recommendation giữ offset và không ghi vector không tương thích.
def _get_embedding_service(request: Request) -> SearchImageEmbeddingService:
    service = getattr(request.app.state, "search_image_embedding_service", None)
    if service is None:
        raise ProviderUnavailableError()
    # Lifespan chỉ gắn service đã khởi tạo; cast ràng buộc kiểu sau khi đã xử lý trạng thái unavailable.
    return cast(SearchImageEmbeddingService, service)


# Chỉ client nội bộ có token được chọn cách tạo khung; caller catalog cũ giữ mặc định center crop.
# Giữ bytes trong RAM và trả 503 khi inference lỗi để Product dùng fallback hiện có.
@router.post("", response_model=SearchImageEmbeddingResponse, response_model_by_alias=True, response_model_exclude_none=True)
async def embed_image(
    request: Request,
    _: Annotated[None, Depends(require_internal_image_token)],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[SearchImageEmbeddingService, Depends(_get_embedding_service)],
    framing: Annotated[ImageFraming, Query(description="Framing used before PE-Core inference.")] = "center_crop",
) -> SearchImageEmbeddingResponse:
    """Kiểm tra MIME, đọc tối đa cấu hình byte cap và chạy ONNX ngoài event loop."""

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in _SUPPORTED_CONTENT_TYPES:
        raise InvalidInputError()
    raw = await _read_bounded_body(request, settings.image_search_max_bytes)
    try:
        # Storefront chuẩn hóa và tái dùng ảnh đã decode; worker catalog tiếp tục lấy một vector kiểu cũ.
        if framing == "preserve_frame":
            query = await run_in_threadpool(service.embed_query_image, raw, content_type)
        else:
            vector = await run_in_threadpool(service.embed_image, raw, content_type, framing)
            query = QueryImageEmbedding(vector)
    except InvalidImageData as error:
        raise InvalidInputError() from error
    except Exception as error:
        # Chỉ ghi mã lỗi nội bộ, không log bytes, filename, owner hoặc vector ảnh.
        logger.warning("Image embedding inference failed: %s", error.__class__.__name__)
        raise ProviderUnavailableError() from error
    return SearchImageEmbeddingResponse(
        model_version=service.model_version,
        dimensions=service.dimensions,
        preprocess_version=service.preprocess_version,
        vector=query.vector,
        detail_vectors=query.detail_vectors,
        rotation_vectors=query.rotation_vectors,
        strong_match_required=query.strong_match_required or None,
    )


@router.get("/status", response_model=SearchImageEmbeddingStatusResponse, response_model_by_alias=True)
async def image_embedding_status(
    _: Annotated[None, Depends(require_internal_image_token)],
    request: Request,
) -> SearchImageEmbeddingStatusResponse:
    """Cho worker xác nhận artifact model/dimension đã sẵn sàng trước khi subscribe/backfill."""

    service = getattr(request.app.state, "search_image_embedding_service", None)
    return SearchImageEmbeddingStatusResponse(
        ready=service is not None,
        model_version=service.model_version if service is not None else None,
        dimensions=service.dimensions if service is not None else None,
        preprocess_version=IMAGE_PREPROCESS_VERSION,
    )
