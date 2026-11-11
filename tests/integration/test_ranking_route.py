"""Kiểm thử HTTP nội bộ cho trạng thái và prediction LambdaRank."""

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import create_app


# Model giả chỉ dùng để xác nhận response contract mà không cần file LightGBM thật.
class FakeRankingModel:
    """Đại diện artifact ranker đã nạp thành công."""

    model_version = "ranking-lambdarank-test-v1"

    def predict(self, rows: list[list[float]]) -> list[float]:
        """Trả điểm raw cùng thứ tự candidate gửi lên."""

        return [-0.25 + index for index, _ in enumerate(rows)]


# Cấu hình test dùng token riêng và không để app đọc env hay artifact của máy.
def _configure_test_app(model: object | None):
    """Tạo app cô lập cho một lần gọi ranking endpoint."""

    application = create_app()
    application.state.ranking_model = model
    application.dependency_overrides[get_settings] = lambda: Settings(
        internal_service_token=SecretStr("internal-test-token")
    )
    return application


# Endpoint status yêu cầu token và chỉ mô tả readiness/version/schema model.
@pytest.mark.asyncio
async def test_ranking_status_requires_token_and_reports_ready_model() -> None:
    """Status không lộ artifact path, score hoặc thông tin không cần thiết."""

    application = _configure_test_app(FakeRankingModel())
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            unauthorized = await client.get("/api/v1/ranking/status")
            response = await client.get(
                "/api/v1/ranking/status",
                headers={"x-internal-service-token": "internal-test-token"},
            )
    finally:
        application.dependency_overrides.clear()

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "modelVersion": "ranking-lambdarank-test-v1",
        "featureCount": 9,
    }


# Model chưa nạp phải được nhận diện unavailable để serving không dùng score khác thay thế.
@pytest.mark.asyncio
async def test_ranking_status_reports_unavailable_without_model() -> None:
    """Status thiếu model không gắn nhãn fallback hoặc modelVersion giả."""

    application = _configure_test_app(None)
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/v1/ranking/status",
                headers={"x-internal-service-token": "internal-test-token"},
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "modelVersion": None,
        "featureCount": 9,
    }


# Predict không có artifact phải trả 503 thay vì chạy bất kỳ công thức serving cũ nào.
@pytest.mark.asyncio
async def test_ranking_prediction_fails_closed_without_model() -> None:
    """Request prediction dừng rõ ràng khi registry chưa có LambdaRank."""

    application = _configure_test_app(None)
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/ranking/predict",
                headers={"x-internal-service-token": "internal-test-token"},
                json={
                    "requestId": "request-1",
                    "items": [{"itemId": "product-1", "features": [0.0] * 9}],
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["code"] == "RANKING_MODEL_UNAVAILABLE"


# Batch raw score giữ được giá trị âm/dương và map theo đúng candidate input.
@pytest.mark.asyncio
async def test_ranking_prediction_returns_raw_scores_in_input_order() -> None:
    """API không ép relevance score thành xác suất."""

    application = _configure_test_app(FakeRankingModel())
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/ranking/predict",
                headers={"x-internal-service-token": "internal-test-token"},
                json={
                    "requestId": "request-1",
                    "items": [
                        {"itemId": "product-a", "features": [0.0] * 9},
                        {"itemId": "product-b", "features": [0.0] * 9},
                    ],
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "requestId": "request-1",
        "modelVersion": "ranking-lambdarank-test-v1",
        "predictions": [
            {"itemId": "product-a", "score": -0.25},
            {"itemId": "product-b", "score": 0.75},
        ],
    }
