"""Media Service client cho download, upload và cleanup ảnh AI.

Worker không truy cập S3 trực tiếp. Adapter giới hạn payload và chỉ chuyển metadata
cần thiết qua internal endpoints được bảo vệ bằng service token.
"""

import base64
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.application.ports import GeneratedImage
from app.modules.image_optimization.domain.models import GeneratedAsset
from app.modules.image_optimization.infrastructure.clients.base import InternalHttpClient


# Kết nối worker với lifecycle asset do Media Service sở hữu.
class HttpMediaAssetClient:
    """Không xóa source asset và không nhận public URL tùy ý từ frontend."""

    # Nhận shared HTTP client để download/upload nhiều ảnh trong cùng connection pool.
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Lưu base URL và helper internal auth."""

        self._base_url = settings.media_service_url.rstrip("/")
        self._http = InternalHttpClient(settings.internal_service_token, client)

    # Tải source bytes qua Media Service đã kiểm tra purpose/ownership.
    async def download_source(self, *, seller_owner_id: UUID, asset_id: UUID) -> tuple[bytes, str, str]:
        """Từ chối response rỗng hoặc upstream status lỗi."""

        try:
            response = await self._http.request(
                "GET",
                f"{self._base_url}/api/v1/media/assets/internal/assets/{asset_id}/download",
                timeout=20,
                params={"purpose": "product_image"},
                headers=self._http.headers(seller_owner_id),
            )
            self._http.ensure_success(response)
            if not response.content:
                raise ProviderUnavailableError()
            return response.content, response.headers.get("content-type", "application/octet-stream"), f"{asset_id}.source"
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    # Upload binary tạm và nhận asset identity do Media Service cấp.
    async def upload_output(self, *, seller_owner_id: UUID, job_id: UUID, output: GeneratedImage) -> GeneratedAsset:
        """Không giữ binary sau khi request hoàn tất."""

        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/api/v1/media/assets/internal/ai-assets/upload",
                timeout=30,
                headers=self._http.headers(seller_owner_id),
                json={
                    "sellerOwnerId": str(seller_owner_id),
                    "jobId": str(job_id),
                    "contentBase64": base64.b64encode(output.content).decode("ascii"),
                    "contentType": output.content_type,
                    "fileName": output.file_name,
                },
            )
            self._http.ensure_success(response)
            body = response.json()
            return GeneratedAsset(
                asset_id=UUID(str(body["assetId"])),
                public_url=body.get("publicUrl"),
                mode="PENDING",
            )
        except (httpx.HTTPError, TimeoutError, KeyError, ValueError, TypeError) as error:
            raise ProviderUnavailableError() from error

    # Yêu cầu cleanup output prefix của job, không đụng source assets.
    async def cleanup_outputs(self, *, seller_owner_id: UUID, job_id: UUID) -> None:
        """Raise lỗi để use case/cleanup worker có thể ghi nhận và retry bền vững."""

        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/api/v1/media/assets/internal/ai-assets/{job_id}/cleanup",
                timeout=20,
                headers=self._http.headers(seller_owner_id),
            )
            self._http.ensure_success(response)
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error
