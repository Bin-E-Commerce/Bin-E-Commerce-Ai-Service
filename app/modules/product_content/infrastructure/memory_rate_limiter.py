"""Rate limiter sliding-window theo seller, dùng memory cho local/test."""

import asyncio
import time
from collections import defaultdict, deque

from app.core.errors import RateLimitExceededError


# Adapter local; production có thể thay bằng Redis atomic counter để đồng bộ nhiều instance.
class MemoryRateLimiter:
    """Giới hạn request seller bằng các timestamp trong cửa sổ trượt."""

    # Mỗi key có deque riêng và lock chung để request đồng thời không vượt quota.
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    # Xóa timestamp cũ, kiểm tra quota hiện tại rồi thêm request mới nếu còn chỗ.
    # Toàn bộ sliding-window operation nằm trong lock để request đồng thời không cùng vượt qua quota.
    # Retry-After được tính từ timestamp cũ nhất, giúp frontend chờ đúng lúc thay vì retry dồn dập gây tốn phí.
    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Giữ timestamp trong cửa sổ trượt và chặn request thứ vượt quá limit."""

        now = time.monotonic()
        async with self._lock:
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= now - window_seconds:
                timestamps.popleft()
            if len(timestamps) >= limit:
                retry_after = max(1, int(window_seconds - (now - timestamps[0])))
                raise RateLimitExceededError(retry_after)
            timestamps.append(now)
