"""Redis sliding-window rate limiter dùng chung giữa các AI capability.

Lua script thực hiện remove/count/add atomically để nhiều instance không cùng vượt
quota. Redis key chỉ chứa hash/UUID, không chứa email, prompt hoặc payload.
"""

import time
from collections.abc import Awaitable
from typing import cast
from uuid import uuid4

from redis.asyncio import Redis

from app.core.errors import RateLimitExceededError

_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  if oldest[2] then return {0, math.max(1, window - (now - tonumber(oldest[2])))} end
  return {0, window}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window)
return {1, 0}
"""


# Giới hạn request trên toàn cluster bằng sorted set theo millisecond.
class RedisRateLimiter:
    """Raise public 429 với Retry-After chính xác khi seller hết quota."""

    # Nhận Redis client từ lifespan; adapter không sở hữu connection lifecycle.
    def __init__(self, client: Redis) -> None:
        """Giữ client đã có pool để mọi request tái sử dụng connection."""

        self._client = client

    # Chạy Lua atomically để tránh race condition giữa nhiều API instance.
    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Thêm request khi còn quota hoặc raise RateLimitExceededError."""

        now_ms = int(time.time() * 1000)
        window_ms = max(1, window_seconds) * 1000
        operation = cast(
            Awaitable[list[object]],
            self._client.eval(
                _SLIDING_WINDOW_SCRIPT,
                1,
                key,
                str(now_ms),
                str(window_ms),
                str(max(1, limit)),
                f"{now_ms}:{uuid4()}",
            ),
        )
        result = await operation
        allowed = cast(int, result[0]) == 1
        if not allowed:
            retry_after_ms = cast(int, result[1])
            raise RateLimitExceededError(max(1, (retry_after_ms + 999) // 1000))
