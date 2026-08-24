"""Khởi tạo FastAPI app, lifecycle adapter và mapping lỗi HTTP ổn định."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError
from app.core.logging import configure_logging
from app.modules.product_content.infrastructure.memory_cache import MemoryResultCache
from app.modules.product_content.infrastructure.memory_rate_limiter import (
    MemoryRateLimiter,
)
from app.modules.product_content.presentation.router import router as product_content_router


# Khởi tạo cache/rate limiter theo lifecycle để không có mutable singleton gọi ngầm khi import module.
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo các adapter nhẹ theo vòng đời process, không gọi mạng khi import."""

    configure_logging()
    application.state.result_cache = MemoryResultCache()
    application.state.rate_limiter = MemoryRateLimiter()
    yield


# Dùng app factory để production và test có cùng dependency wiring, tránh state dùng chung ngoài ý muốn.
def create_app() -> FastAPI:
    """Tạo FastAPI app để production và test dùng chung một wiring path."""

    application = FastAPI(
        title="Bin AI Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Chỉ trả mã lỗi công khai; chi tiết exception nội bộ không được gửi ra client.
    @application.exception_handler(AppError)
    async def handle_app_error(_: Request, error: AppError) -> JSONResponse:
        """Map lỗi nội bộ thành payload ổn định, không làm lộ provider detail."""

        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.public_message},
            headers=error.headers,
        )

    # Health check chỉ kiểm tra process sống, không gọi OpenAI để tránh phát sinh chi phí.
    @application.get("/api/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """Trả trạng thái sống của service mà không kiểm tra provider trả phí."""

        return {"status": "ok", "service": "ai-service"}

    application.include_router(product_content_router)
    return application


app = create_app()
