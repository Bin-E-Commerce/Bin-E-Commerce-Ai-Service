"""Fixture dùng chung cho test, bảo đảm provider thật không bị gọi và không phát sinh phí."""

from collections.abc import AsyncIterator

import pytest_asyncio

from app.main import create_app


@pytest_asyncio.fixture
# Tạo app factory riêng cho mỗi test để state cache/rate limit không rò giữa các case.
async def application() -> AsyncIterator[object]:
    """Tạo app độc lập cho từng test; provider thật không được gọi và không phát sinh chi phí."""

    yield create_app()
