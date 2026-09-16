"""Kiểm tra HTTP status model ranking và chặn truy cập không có internal token."""

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import create_app


class FakeRankingModel:
    """Model fake tối giản để test status mà không load artifact thực."""

    model_version = "ranking-lgbm-v1"


@pytest.mark.asyncio
async def test_ranking_status_requires_internal_token_and_reports_ready_model() -> None:
    """Status chỉ cho Recommendation Service nội bộ đọc và trả version model không lộ path."""

    application = create_app()
    application.state.ranking_model = FakeRankingModel()
    application.dependency_overrides[get_settings] = lambda: Settings(internal_service_token=SecretStr("internal-test-token"))
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
        "fallback": False,
        "modelVersion": "ranking-lgbm-v1",
        "featureCount": 9,
    }


@pytest.mark.asyncio
async def test_ranking_status_reports_fallback_model() -> None:
    """Artifact chưa có thì status phải nói rõ fallback để Admin không hiểu nhầm là AI đang chạy."""

    application = create_app()
    application.state.ranking_model = type("FallbackModel", (), {"model_version": "ranking-fallback-v1"})()
    application.dependency_overrides[get_settings] = lambda: Settings(internal_service_token=SecretStr("internal-test-token"))
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/ranking/status",
                headers={"x-internal-service-token": "internal-test-token"},
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["fallback"] is True
