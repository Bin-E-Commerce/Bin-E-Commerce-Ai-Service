"""Kiểm tra HTTP boundary, permission và response mapping bằng service fake."""

import httpx
import pytest

from app.core.config import Settings, get_settings
from app.core.dependencies import get_product_name_service
from app.main import create_app
from app.modules.product_content.domain.models import GeneratedName, SuggestionBatch


# Fake service cô lập route khỏi OpenAI để test không gửi request trả phí.
class FakeService:
    # Trả batch hợp lệ để kiểm tra mapping domain model sang JSON alias camelCase.
    async def generate(self, command, user_id):
        return (
            "request-1",
            SuggestionBatch(
                suggestions=(
                    GeneratedName("Premium Vietnamese leather shoes for office", "Clear category.", True),
                    GeneratedName("Comfortable leather shoes for daily business outfits", "Useful context.", False),
                    GeneratedName("Elegant men's leather shoes with soft sole", "Natural wording.", False),
                ),
                warnings=(),
            ),
        )


# Body mẫu có ảnh CDN hợp lệ và locale đúng contract.
def request_body() -> dict[str, object]:
    return {
        "category": {"name": "Shoes"},
        "brand": "Bin",
        "images": [
            {
                "assetId": "asset-1",
                "publicUrl": "https://cdn.example.com/shoes.jpg",
                "fileName": "shoes.jpg",
            }
        ],
        "locale": "vi-VN",
    }


@pytest.mark.asyncio
# Không có permission thì route phải trả 403 trước khi khởi tạo LLM provider.
async def test_route_requires_ai_permission() -> None:
    application = create_app()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/seller/product-content/name-suggestions",
            json=request_body(),
            headers={"x-user-id": "seller-1"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
# Permission hợp lệ phải trả ba suggestion và requestId mà không đổi field ngoài contract.
async def test_route_returns_structured_suggestions() -> None:
    application = create_app()
    application.dependency_overrides[get_product_name_service] = lambda: FakeService()
    application.dependency_overrides[get_settings] = lambda: Settings(media_public_cdn_url="https://cdn.example.com")
    transport = httpx.ASGITransport(app=application)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/seller/product-content/name-suggestions",
                json=request_body(),
                headers={
                    "x-user-id": "seller-1",
                    "x-user-permissions": "seller.ai.product_content.generate",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body["suggestions"]) == 3
    assert body["suggestions"][0]["recommended"] is True
    assert body["requestId"] == "request-1"
