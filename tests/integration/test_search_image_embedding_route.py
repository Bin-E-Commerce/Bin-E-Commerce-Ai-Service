"""Kiểm thử contract HTTP nội bộ với model giả, không tải artifact hoặc gửi ảnh thật."""

from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.modules.search_ranking.image_embedding.application.image_preprocessing import IMAGE_PREPROCESS_VERSION, ImageFraming
from app.modules.search_ranking.image_embedding.application.service import QueryImageEmbedding
from app.modules.search_ranking.image_embedding.presentation.router import router


# Adapter giả ghi nhận lựa chọn tạo khung để test contract mà không chạy model.
class FakeImageEmbeddingService:
    model_version = "pe-core-route-test-v1"
    dimensions = 512
    preprocess_version = IMAGE_PREPROCESS_VERSION

    # Mô phỏng vector đã normalize và ghi nhận cách tạo khung, không giữ bytes.
    def embed_image(self, raw: bytes, content_type: str, framing: ImageFraming) -> list[float]:
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert content_type == "image/png"
        self.framing = framing
        return [1.0] + [0.0] * 511

    # Mô phỏng hai vector khác nhau để xác minh route chỉ xuất chúng cho query storefront.
    def embed_query_image(self, raw: bytes, content_type: str) -> QueryImageEmbedding:
        full = self.embed_image(raw, content_type, "preserve_frame")
        return QueryImageEmbedding(full, detail_vectors=(full, [0.0, 1.0] + [0.0] * 510), strong_match_required=True)


# Tạo PNG hợp lệ trong memory để test route không dùng ảnh trên ổ đĩa.
def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


# App test chỉ cài token giả và model giả; không đọc environment hoặc gọi inference bên ngoài.
def _configured_application():
    application = FastAPI()
    application.include_router(router)

    # Cùng status/code cơ bản với app factory mà không import module AI không liên quan đang thiếu dependency.
    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"code": error.code})

    application.state.search_image_embedding_service = FakeImageEmbeddingService()
    application.dependency_overrides[get_settings] = lambda: Settings(
        internal_service_token=SecretStr("image-test-token"),
        image_search_max_bytes=1024,
    )
    return application


# Route từ chối request không có token trước khi chuyển bytes vào model.
@pytest.mark.asyncio
async def test_image_embedding_route_requires_internal_token() -> None:
    application = _configured_application()
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search-ranking/image-embedding",
                content=_png_bytes(),
                headers={"content-type": "image/png"},
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 401


# Route trả model version/dimension đúng schema và không nhận form/body multipart có overhead ngoài contract.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_string", "expected_framing"),
    [("", "center_crop"), ("?framing=preserve_frame", "preserve_frame")],
)
async def test_image_embedding_route_returns_vector_for_authenticated_png(query_string: str, expected_framing: str) -> None:
    application = _configured_application()
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search-ranking/image-embedding" + query_string,
                content=_png_bytes(),
                headers={
                    "content-type": "image/png",
                    "x-internal-service-token": "image-test-token",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["modelVersion"] == "pe-core-route-test-v1"
    assert response.json()["dimensions"] == 512
    assert response.json()["preprocessVersion"] == IMAGE_PREPROCESS_VERSION
    assert len(response.json()["vector"]) == 512
    assert application.state.search_image_embedding_service.framing == expected_framing
    # Worker catalog tiếp tục nhận payload cũ, không tốn inference hoặc dung lượng của vector chi tiết.
    if expected_framing == "preserve_frame":
        assert [vector[:2] for vector in response.json()["detailVectors"]] == [[1.0, 0.0], [0.0, 1.0]]
        assert response.json()["strongMatchRequired"] is True
    else:
        assert "detailVectors" not in response.json()
        assert "strongMatchRequired" not in response.json()


# Byte limit được kiểm tra ở HTTP boundary trước inference, kể cả ảnh có đúng MIME.
@pytest.mark.asyncio
async def test_image_embedding_route_rejects_body_over_limit() -> None:
    application = _configured_application()
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search-ranking/image-embedding",
                content=_png_bytes() + (b"0" * 2048),
                headers={
                    "content-type": "image/png",
                    "x-internal-service-token": "image-test-token",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
