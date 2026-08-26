"""HTTP adapters noi bo cho Media Service va Product Service.

Worker va application layer chi phu thuoc protocol domain; file nay la noi duy nhat biet URL,
internal token va payload HTTP cua service khac.
"""

from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.domain.models import GeneratedAsset
from app.modules.image_optimization.domain.ports import GeneratedImage, MediaAssetClient, ProductMediaClient, ProductOwnerClient


class HttpProductOwnerClient(ProductOwnerClient):
    """Doc product qua Product Service de lay ownership va optimistic version luc tao job."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.product_service_url.rstrip("/")
        self._token = settings.internal_service_token

    async def assert_owned_and_get_updated_at(self, seller_owner_id: UUID, product_id: UUID) -> datetime:
        """Product Service la source of truth cho ownership, khong tin product ID tu frontend."""

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._base_url}/api/v1/seller/products/{product_id}",
                    headers={
                        "x-user-id": str(seller_owner_id),
                        "x-user-email": "ai-service@internal.local",
                        # Product Service cần quyền đọc để xác minh ownership và lấy phiên bản sản phẩm trước khi tạo job.
                        "x-user-permissions": "seller.product.read,seller.ai.image_optimization.generate",
                        "x-internal-service-token": self._token.get_secret_value() if self._token else "",
                    },
                )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
            updated_at = response.json().get("updatedAt")
            if not isinstance(updated_at, str):
                raise ProviderUnavailableError()
            return datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError) as error:
            raise ProviderUnavailableError() from error

    async def get_cover_asset_id(self, seller_owner_id: UUID, product_id: UUID) -> UUID:
        """Lay externalImageId cua cover; khong nhan URL asset tu frontend."""

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._base_url}/api/v1/seller/products/{product_id}",
                    headers={
                        "x-user-id": str(seller_owner_id),
                        "x-user-email": "ai-service@internal.local",
                        # Dùng cùng context quyền ở bước lấy ảnh cover, tránh tạo job nhưng không đọc được media nguồn.
                        "x-user-permissions": "seller.product.read,seller.ai.image_optimization.generate",
                        "x-internal-service-token": self._token.get_secret_value() if self._token else "",
                    },
                )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
            images = response.json().get("images")
            cover = next((image for image in images or [] if image.get("isThumbnail")), None)
            asset_id = cover.get("externalImageId") if isinstance(cover, dict) else None
            if not asset_id and isinstance(cover, dict):
                # Hỗ trợ product cũ chưa lưu externalImageId nhưng URL vẫn do Media Service phát hành.
                asset_id = self._asset_id_from_product_url(cover.get("imageUrl"), seller_owner_id)
            if not asset_id:
                raise ProviderUnavailableError()
            return UUID(str(asset_id))
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as error:
            raise ProviderUnavailableError() from error

    # Đọc toàn bộ gallery từ Product Service và chỉ nhận các asset ID xuất hiện trong graph của product.
    async def get_product_asset_ids(
        self, seller_owner_id: UUID, product_id: UUID, requested_asset_ids: tuple[UUID, ...]
    ) -> tuple[UUID, ...]:
        """Xác minh ownership và lựa chọn ảnh, không tin danh sách asset do browser tự gửi."""

        if not requested_asset_ids:
            raise ProviderUnavailableError()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._base_url}/api/v1/seller/products/{product_id}",
                    headers={
                        "x-user-id": str(seller_owner_id),
                        "x-user-email": "ai-service@internal.local",
                        "x-user-permissions": "seller.product.read,seller.ai.image_optimization.generate",
                        "x-internal-service-token": self._token.get_secret_value() if self._token else "",
                    },
                )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
            images = response.json().get("images")
            available = {
                UUID(asset_id)
                for image in images or []
                if isinstance(image, dict)
                for asset_id in [self._asset_id_from_product_url(image.get("imageUrl"), seller_owner_id)]
                if asset_id
            }
            selected = tuple(asset_id for asset_id in requested_asset_ids if asset_id in available)
            if len(selected) != len(requested_asset_ids):
                raise ProviderUnavailableError()
            return selected
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as error:
            raise ProviderUnavailableError() from error

    # Chuyển URL media chuẩn về asset ID, đồng thời chặn URL không thuộc seller hiện tại.
    @staticmethod
    def _asset_id_from_product_url(media_url: object, seller_owner_id: UUID) -> str | None:
        """Chỉ nhận HTTPS URL có path processed product_image và UUID hợp lệ."""

        if not isinstance(media_url, str) or not media_url.startswith("https://"):
            return None
        parts = [part for part in urlparse(media_url).path.split("/") if part]
        if len(parts) < 5 or parts[:3] != ["media", "processed", "product_image"]:
            return None
        if parts[3].lower() != str(seller_owner_id).lower():
            return None
        try:
            return str(UUID(parts[4]))
        except ValueError:
            return None


# Adapter HTTP dùng connection pool chung để giảm handshake khi worker xử lý batch.
class HttpMediaAssetClient(MediaAssetClient):
    """Goi Media Service de doc source/upload output/cleanup theo internal token."""

    # Nhận client dùng chung từ worker; nếu không truyền thì vẫn hỗ trợ adapter độc lập trong test.
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = settings.media_service_url.rstrip("/")
        self._token = settings.internal_service_token
        self._client = client

    # Tải source qua Media Service với connection pool và allow-list purpose đã có sẵn.
    async def download_source(self, *, seller_owner_id: UUID, asset_id: UUID) -> tuple[bytes, str, str]:
        """Tai anh product_image qua endpoint allow-list purpose."""

        try:
            response = await self._request(
                "GET",
                f"{self._base_url}/api/v1/media/assets/internal/assets/{asset_id}/download",
                timeout=20,
                params={"purpose": "product_image"},
                headers=self._headers(seller_owner_id),
            )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
            return response.content, response.headers.get("content-type", "application/octet-stream"), f"{asset_id}.source"
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    # Upload output qua Media Service mà không cho worker truy cập trực tiếp S3.
    async def upload_output(self, *, seller_owner_id: UUID, job_id: UUID, output: GeneratedImage) -> GeneratedAsset:
        """Encode output chi trong request noi bo, Media Service gioi han kich thuoc va tao asset ID."""

        import base64

        try:
            response = await self._request(
                "POST",
                f"{self._base_url}/api/v1/media/assets/internal/ai-assets/upload",
                timeout=30,
                headers=self._headers(seller_owner_id),
                json={
                    "sellerOwnerId": str(seller_owner_id),
                    "jobId": str(job_id),
                    "contentBase64": base64.b64encode(output.content).decode("ascii"),
                    "contentType": output.content_type,
                    "fileName": output.file_name,
                },
            )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
            body = response.json()
            return GeneratedAsset(asset_id=UUID(body["assetId"]), public_url=body.get("publicUrl"), mode=output.file_name)
        except (httpx.HTTPError, TimeoutError, KeyError, ValueError, TypeError) as error:
            raise ProviderUnavailableError() from error

    # Dọn output tạm theo job, tuyệt đối không chạm asset gốc của sản phẩm.
    async def cleanup_outputs(self, *, seller_owner_id: UUID, job_id: UUID) -> None:
        """Don output AI theo job, source original khong nam trong prefix nay."""

        try:
            response = await self._request(
                "POST",
                f"{self._base_url}/api/v1/media/assets/internal/ai-assets/{job_id}/cleanup",
                timeout=20,
                headers=self._headers(seller_owner_id),
            )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    def _headers(self, owner_id: UUID) -> dict[str, str]:
        """Tao header noi bo toi Media Service ma khong ghi token ra log."""

        return {"x-user-id": str(owner_id), "x-internal-service-token": self._token.get_secret_value() if self._token else ""}

    # Dùng connection pool của worker khi có client chung; local/test vẫn tự đóng client tạm thời.
    async def _request(self, method: str, url: str, *, timeout: float, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            return await self._client.request(method, url, timeout=timeout, **kwargs)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, **kwargs)


class HttpProductMediaClient(ProductMediaClient):
    """Goi Product Service voi timeout ngan va khong log payload co URL media."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.product_service_url.rstrip("/")
        self._token = settings.internal_service_token

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
        """Forward apply sau khi seller da xem preview; Product Service van check ownership lan cuoi."""

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
        headers = self._headers(seller_owner_id, permissions)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/seller/products/{product_id}/ai-media/apply",
                    json=payload,
                    headers=headers,
                )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    async def rollback_media(self, *, seller_owner_id: UUID, product_id: UUID, job_id: UUID) -> None:
        """Forward rollback de Product Service khoi phuc snapshot trong transaction."""

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/seller/products/{product_id}/ai-media/rollback",
                    json={"jobId": str(job_id)},
                    headers=self._headers(seller_owner_id, ("seller.ai.image_optimization.rollback",)),
                )
            if response.status_code >= 400:
                raise ProviderUnavailableError()
        except (httpx.HTTPError, TimeoutError) as error:
            raise ProviderUnavailableError() from error

    def _headers(self, owner_id: UUID, permissions: tuple[str, ...]) -> dict[str, str]:
        """Tao header context toi thieu; khong forward JWT hoac secret cua client."""

        return {
            "content-type": "application/json",
            "x-user-id": str(owner_id),
            "x-user-email": "ai-service@internal.local",
            "x-user-permissions": ",".join(permissions),
            "x-internal-service-token": self._token.get_secret_value() if self._token else "",
        }
