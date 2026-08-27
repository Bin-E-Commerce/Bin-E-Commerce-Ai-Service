"""Port cache và rate limit dùng chung cho các product-content use case."""

from typing import Protocol

from app.modules.product_content.domain.models import DescriptionBatch, SuggestionBatch


# Cache chỉ lưu kết quả đã sanitize, không lưu prompt hoặc provider response raw.
class ResultCache(Protocol):
    """Hợp đồng cache TTL cho product content."""

    # Đọc fingerprint không chứa raw payload.
    async def get(self, key: str) -> SuggestionBatch | DescriptionBatch | None:
        """Trả kết quả đã kiểm duyệt hoặc None."""

    # Ghi kết quả đã kiểm duyệt với TTL bắt buộc.
    async def set(self, key: str, value: SuggestionBatch | DescriptionBatch, ttl_seconds: int) -> None:
        """Không cho cache vô thời hạn."""


# Rate limit chạy sau cache để request lặp hợp lệ không tiêu quota thêm.
class RateLimiter(Protocol):
    """Hợp đồng giới hạn request theo seller."""

    # Raise lỗi ổn định khi quota cửa sổ đã hết.
    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Kiểm tra quota bằng adapter memory hoặc Redis atomic."""
