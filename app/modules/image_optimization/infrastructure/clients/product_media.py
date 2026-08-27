"""Product Service client cho apply và rollback ảnh AI.

Adapter chỉ gửi output đã được use case xác minh. Product Service tiếp tục kiểm tra
ownership, optimistic version và job ID idempotency trước khi mutation.
"""

from datetime import datetime
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.domain.models import GeneratedAsset
from app.modules.image_optimization.infrastructure.clients.base import InternalHttpClient


# Gọi mutation nội bộ của Product Service bằng shared HTTP client.
class HttpProductMediaClient:
    """Không tự tạo permission và không tin URL từ browser."""

    # Nhận settings và optional connection pool từ lifespan/worker.
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Chuẩn hóa base URL và internal auth helper."""

        self._base_url = settings.product_service_url.rstrip("/")
        self._http = InternalHttpClient(settings.internal_service_token, client)

    # Áp dụng đúng output seller chọn với optimistic product version.
    async def apply_media(
        self,
        *,
        seller_owner_id: UUID,
        product_id: UUID,
        job_id: UUID,
        expected_product_updated_at: datetime | None,
        assets: tuple[GeneratedAsset, ...],
        permissions: tuple[str, ...] = (),
    ) -> None:
        """Job ID là idempotency identity cho downstream retry sau partial failure."""

        if expected_product_updated_at is None:
            raise ValueError("Expected product version is required")
        payload = {
            "jobId": str(job_id),
            "expectedProductUpdatedAt": expected_product_updated_at.isoformat(),
            "images": [
                {"assetId": str(asset.asset_id), "imageUrl": asset.public_url, "sortOrder": index}
                for index, asset in enumerate(assets)
                if asset.public_url
            ],
        }
        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/api/v1/seller/products/{product_id}/ai-media/apply",
                timeout=15,
                json=payload,
                headers=self._http.headers(seller_owner_id, frozenset(permissions)),
            )
            self._http.ensure_success(response)
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    # Yêu cầu Product Service khôi phục snapshot ảnh gốc theo job ID.
    async def rollback_media(
        self,
        *,
        seller_owner_id: UUID,
        product_id: UUID,
        job_id: UUID,
        permissions: tuple[str, ...] = (),
    ) -> None:
        """Không synthesize permission; service token và ownership vẫn được downstream kiểm tra."""

        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/api/v1/seller/products/{product_id}/ai-media/rollback",
                timeout=15,
                json={"jobId": str(job_id)},
                headers=self._http.headers(seller_owner_id, frozenset(permissions)),
            )
            self._http.ensure_success(response)
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error
