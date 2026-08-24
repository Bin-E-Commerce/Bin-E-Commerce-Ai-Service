"""Protocol domain cho LLM provider, cache và rate limiter có thể thay thế."""

from collections.abc import Sequence
from typing import Protocol

from app.modules.product_content.domain.models import GeneratedName, ProductContext, SuggestionBatch


# Port này giúp use case không phụ thuộc OpenAI; Anthropic/Gemini/local model chỉ cần implement protocol.
class LLMNameSuggestionProvider(Protocol):
    """Hợp đồng sinh tên sản phẩm cho mọi LLM adapter."""

    # Provider nhận domain context và trả ứng viên, không được tự đọc HTTP header hay cache.
    async def generate_name_suggestions(self, context: ProductContext) -> Sequence[GeneratedName]:
        """Sinh ứng viên mà không biết chi tiết HTTP, auth, cache hoặc persistence."""


# Cache port tách khỏi use case để local dùng memory, production có thể thay Redis.
class ResultCache(Protocol):
    """Hợp đồng cache response đã được sanitize và giới hạn TTL."""

    # Đọc cache theo fingerprint, không tìm theo nội dung thô hoặc user secret.
    async def get(self, key: str) -> SuggestionBatch | None:
        """Đọc kết quả cache không chứa dữ liệu nhạy cảm."""

    # Ghi kết quả có TTL để tránh cache vô thời hạn dữ liệu product content.
    async def set(self, key: str, value: SuggestionBatch, ttl_seconds: int) -> None:
        """Lưu kết quả với thời hạn hết hạn rõ ràng."""


# Rate limiter port bảo vệ quota theo seller và cho phép thay memory bằng Redis atomic counter.
class RateLimiter(Protocol):
    """Hợp đồng giới hạn request theo seller trong cửa sổ thời gian."""

    # Hàm phải raise lỗi ổn định khi caller vượt quota thay vì âm thầm bỏ qua.
    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Từ chối khi caller đã dùng hết quota trong cửa sổ cấu hình."""
