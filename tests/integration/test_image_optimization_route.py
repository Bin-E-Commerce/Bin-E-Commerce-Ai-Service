"""Kiem thu HTTP contract image optimization, permission va idempotency boundary."""

from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from app.bootstrap.dependencies import get_image_optimization_service
from app.main import create_app
from app.modules.image_optimization.domain.enums import ImageOptimizationMode, ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob


class FakeImageService:
    """Fake application service de route test khong can PostgreSQL/Kafka."""

    async def create_jobs(self, command):
        """Tra mot job deterministic theo command da parse."""

        job = ImageOptimizationJob.create(
            seller_owner_id=command.seller_owner_id,
            product_id=command.product_ids[0],
            source_asset_ids=(uuid4(),),
            requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
            idempotency_key=command.idempotency_key,
            expected_product_updated_at=datetime.now(UTC),
        )
        return command.idempotency_key, (job,)


def body() -> dict[str, object]:
    """Body camelCase dung dung voi contract public."""

    return {
        "productIds": [str(uuid4())],
        "sourceAssetPolicy": "COVER_IMAGE",
        "modes": ["WHITE_BACKGROUND"],
    }


@pytest.mark.asyncio
async def test_image_optimization_requires_generate_permission() -> None:
    """AAA: request thieu permission phai bi chan truoc use case."""

    application = create_app()
    application.dependency_overrides[get_image_optimization_service] = lambda: FakeImageService()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/seller/ai/image-optimization/jobs", json=body(), headers={"x-user-id": str(uuid4())}
        )

    assert response.status_code == 403
    application.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_image_optimization_returns_accepted_job() -> None:
    """AAA: permission va idempotency key hop le tra HTTP 202 camelCase."""

    application = create_app()
    application.dependency_overrides[get_image_optimization_service] = lambda: FakeImageService()
    transport = httpx.ASGITransport(app=application)
    owner_id = uuid4()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/seller/ai/image-optimization/jobs",
            json=body(),
            headers={
                "x-user-id": str(owner_id),
                "x-user-permissions": "seller.ai.image_optimization.generate",
                "Idempotency-Key": "image-optimization-route-test",
            },
        )

    assert response.status_code == 202
    assert response.json()["batchId"] == "image-optimization-route-test"
    assert response.json()["jobs"][0]["status"] == ImageOptimizationStatus.PENDING.value
    assert response.json()["jobs"][0]["expectedProductUpdatedAt"]
    application.dependency_overrides.clear()
