"""Cache memory theo process cho local/test, có cùng contract với Redis adapter tương lai."""

import asyncio
import time
from dataclasses import dataclass

from app.modules.product_content.domain.models import DescriptionBatch, SuggestionBatch


# Entry giữ thời điểm hết hạn để cache không trả dữ liệu quá TTL đã cam kết.
@dataclass(frozen=True)
class _CacheEntry:
    value: SuggestionBatch | DescriptionBatch
    expires_at: float


# Adapter đơn giản cho local; production có thể thay bằng Redis mà không sửa application service.
class MemoryResultCache:
    """Cache kết quả đã sanitize trong memory với TTL rõ ràng."""

    # Khởi tạo dict và lock để hai request đồng thời không làm hỏng trạng thái cache.
    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    # Đọc cache và xóa entry hết hạn ngay trong lock để không trả dữ liệu stale.
    # Lock bao trùm cả bước đọc và dọn dẹp để hai coroutine không cùng trả một entry vừa hết hạn.
    # Monotonic clock được dùng thay vì wall-clock nhằm tránh thay đổi giờ hệ thống làm sai TTL.
    async def get(self, key: str) -> SuggestionBatch | DescriptionBatch | None:
        """Đọc cache và loại entry hết hạn trong cùng một critical section."""

        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None
            return entry.value

    # Ghi value đã validate cùng TTL hữu hạn, tránh lưu prompt hoặc dữ liệu thô.
    # Thời điểm hết hạn được tính bằng monotonic clock và chỉ lưu batch đã sanitize, không lưu request/prompt.
    # Adapter production có thể thay bằng Redis nhưng vẫn phải giữ contract TTL và dữ liệu tối thiểu này.
    async def set(self, key: str, value: SuggestionBatch | DescriptionBatch, ttl_seconds: int) -> None:
        """Lưu response đã sanitize với TTL hữu hạn."""

        async with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=time.monotonic() + ttl_seconds)
