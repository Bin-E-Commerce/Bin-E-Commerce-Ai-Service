"""Kiểm tra seller không thể vượt rate limit trong cùng một cửa sổ thời gian."""

import pytest

from app.core.errors import RateLimitExceededError
from app.modules.product_content.infrastructure.memory_rate_limiter import MemoryRateLimiter


@pytest.mark.asyncio
# Request thứ hai phải bị chặn để kiểm soát quota và chi phí LLM theo seller.
async def test_rate_limiter_rejects_request_after_limit() -> None:
    limiter = MemoryRateLimiter()

    await limiter.check("seller-1", limit=1, window_seconds=600)
    with pytest.raises(RateLimitExceededError):
        await limiter.check("seller-1", limit=1, window_seconds=600)
