"""HTTP primitive dùng chung cho internal service clients.

File quản lý connection reuse và service token; không chứa route hoặc payload
nghiệp vụ của Product/Media Service.
"""

from typing import Any
from uuid import UUID

import httpx
from pydantic import SecretStr

from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    InvalidInputError,
    ProviderUnavailableError,
    RateLimitExceededError,
    ResourceNotFoundError,
    UpstreamConflictError,
    UpstreamRequestError,
)


# Cung cấp request helper có thể dùng shared AsyncClient từ application lifespan.
class InternalHttpClient:
    """Không log URL, token, payload hoặc user context khi upstream lỗi."""

    # Nhận optional client; adapter test có thể dùng MockTransport qua client truyền vào.
    def __init__(self, token: SecretStr | None, client: httpx.AsyncClient | None = None) -> None:
        """Lưu secret dạng SecretStr và không chuyển thành plain text trước khi tạo header."""

        self._token = token
        self._client = client

    # Tạo context header từ permission đã được Gateway xác thực, không tự bịa permission.
    def headers(self, owner_id: UUID, permissions: frozenset[str] = frozenset()) -> dict[str, str]:
        """Không forward JWT, email hoặc header ngoài allow-list."""

        return {
            "x-user-id": str(owner_id),
            "x-user-permissions": ",".join(sorted(permissions)),
            "x-internal-service-token": self._token.get_secret_value() if self._token else "",
        }

    # Dùng connection pool chung; fallback client tạm chỉ dành cho unit test adapter độc lập.
    async def request(self, method: str, url: str, *, timeout: float, **kwargs: Any) -> httpx.Response:
        """Trả response để adapter nghiệp vụ tự map status và schema."""

        if self._client is not None:
            return await self._client.request(method, url, timeout=timeout, **kwargs)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, **kwargs)

    # Map status theo semantics thay vì gom mọi upstream error thành provider unavailable 503.
    @staticmethod
    def ensure_success(response: httpx.Response) -> None:
        """Không đọc hoặc trả raw error body vì downstream có thể chứa chi tiết nội bộ."""

        if response.status_code < 400:
            return
        if response.status_code == 400:
            raise UpstreamRequestError()
        if response.status_code == 401:
            raise AuthenticationError()
        if response.status_code == 403:
            raise AuthorizationError()
        if response.status_code == 404:
            raise ResourceNotFoundError()
        if response.status_code == 409:
            raise UpstreamConflictError()
        if response.status_code == 422:
            raise InvalidInputError()
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after", "1")
            raise RateLimitExceededError(int(retry_after) if retry_after.isdigit() else 1)
        if response.status_code >= 500:
            raise ProviderUnavailableError()
        raise UpstreamRequestError()
