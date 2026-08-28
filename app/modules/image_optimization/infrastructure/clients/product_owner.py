"""Product Service client cho ownership, version và source asset selection.

Adapter không thay đổi sản phẩm và không tự tạo permission; nó chỉ chuyển context
đã được Gateway xác thực và map response sang primitive application cần.
"""

from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.errors import InvalidInputError, ProviderUnavailableError
from app.modules.image_optimization.infrastructure.clients.base import InternalHttpClient


# Đọc Product Service như source of truth cho ownership và gallery assets.
class HttpProductOwnerClient:
    """Từ chối source asset không nằm trong product graph của seller."""

    # Nhận shared HTTP client từ lifespan để một batch không tạo nhiều TCP connection.
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Chuẩn hóa base URL và giữ token trong helper chung."""

        self._base_url = settings.product_service_url.rstrip("/")
        self._http = InternalHttpClient(settings.internal_service_token, client)

    # Xác minh product và trả optimistic version dùng ở apply.
    async def assert_owned_and_get_updated_at(
        self, seller_owner_id: UUID, product_id: UUID, permissions: frozenset[str] = frozenset(), seller_email: str = ""
    ) -> datetime:
        """Không tin product ID từ frontend trước khi tạo job trả phí."""

        body = await self._get_product(seller_owner_id, product_id, permissions, seller_email)
        updated_at = body.get("updatedAt")
        if not isinstance(updated_at, str):
            raise ProviderUnavailableError()
        try:
            return datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ProviderUnavailableError() from error

    # Lấy asset ảnh đại diện đã được Product Service trả về cho đúng owner.
    async def get_cover_asset_id(
        self, seller_owner_id: UUID, product_id: UUID, permissions: frozenset[str] = frozenset(), seller_email: str = ""
    ) -> UUID:
        """Không nhận URL hoặc asset cover trực tiếp từ browser."""

        body = await self._get_product(seller_owner_id, product_id, permissions, seller_email)
        images = body.get("images")
        cover = next((item for item in images or [] if isinstance(item, dict) and item.get("isThumbnail")), None)
        if not isinstance(cover, dict):
            raise InvalidInputError()
        # Ưu tiên asset nguồn bất biến; externalImageId chỉ là fallback cho dữ liệu legacy chưa migrate.
        # Tối ưu lại phải đọc ảnh đang hiển thị (kể cả output AI trước đó), sau đó mới fallback về ảnh gốc.
        asset_id = (
            cover.get("aiAssetId")
            or cover.get("sourceAssetId")
            or cover.get("externalImageId")
            or self._asset_id_from_product_url(cover.get("imageUrl"), seller_owner_id)
        )
        try:
            return UUID(str(asset_id))
        except (ValueError, TypeError) as error:
            raise InvalidInputError() from error

    # Đối chiếu toàn bộ asset seller chọn với gallery hiện tại của product.
    async def get_product_asset_ids(
        self,
        seller_owner_id: UUID,
        product_id: UUID,
        requested_asset_ids: tuple[UUID, ...],
        permissions: frozenset[str] = frozenset(),
        seller_email: str = "",
    ) -> tuple[UUID, ...]:
        """Giữ thứ tự seller chọn nhưng từ chối cả request nếu có một asset không thuộc product."""

        if not requested_asset_ids or len(set(requested_asset_ids)) != len(requested_asset_ids):
            raise InvalidInputError()
        body = await self._get_product(seller_owner_id, product_id, permissions, seller_email)
        available: set[UUID] = set()
        for image in body.get("images") or []:
            if not isinstance(image, dict):
                continue
            # Mỗi output AI vẫn phải được resolve về asset nguồn của chính product image row.
            # Dùng output đang active để seller có thể tạo một phiên bản lifestyle mới từ ảnh đã tối ưu.
            raw_asset_id = (
                image.get("aiAssetId")
                or image.get("sourceAssetId")
                or image.get("externalImageId")
                or self._asset_id_from_product_url(image.get("imageUrl"), seller_owner_id)
            )
            try:
                available.add(UUID(str(raw_asset_id)))
            except (ValueError, TypeError):
                continue
        if any(asset_id not in available for asset_id in requested_asset_ids):
            raise InvalidInputError()
        return requested_asset_ids

    # Gọi endpoint seller detail với context thật đã xác thực.
    async def _get_product(
        self,
        seller_owner_id: UUID,
        product_id: UUID,
        permissions: frozenset[str],
        seller_email: str = "",
    ) -> dict[str, Any]:
        """Map transport/schema lỗi thành public provider error không lộ response body."""

        try:
            response = await self._http.request(
                "GET",
                f"{self._base_url}/api/v1/seller/products/{product_id}",
                timeout=10,
                headers=self._http.headers(seller_owner_id, permissions, seller_email),
            )
            self._http.ensure_success(response)
            body = response.json()
            if not isinstance(body, dict):
                raise ProviderUnavailableError()
            return body
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError) as error:
            raise ProviderUnavailableError() from error

    # Suy asset ID từ URL media legacy và khóa path vào đúng owner hiện tại.
    @staticmethod
    def _asset_id_from_product_url(media_url: object, seller_owner_id: UUID) -> str | None:
        """Chỉ nhận HTTPS processed product_image path có UUID hợp lệ."""

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
