"""Khởi tạo FastAPI app, lifecycle adapter và mapping lỗi HTTP ổn định."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.modules.image_optimization.infrastructure.publisher import InMemoryOptimizationEventPublisher
from app.modules.image_optimization.infrastructure.repository import InMemoryImageOptimizationJobRepository
from app.modules.image_optimization.presentation.router import router as image_optimization_router
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
    application.state.image_optimization_repository = InMemoryImageOptimizationJobRepository()
    application.state.image_optimization_publisher = InMemoryOptimizationEventPublisher()
    settings = get_settings()
    if settings.database_url:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        application.state.image_session_factory = async_sessionmaker(engine, expire_on_commit=False)
        application.state.image_database_engine = engine
    yield
    database_engine = getattr(application.state, "image_database_engine", None)
    if database_engine is not None:
        await database_engine.dispose()


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
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        """Map lỗi nội bộ thành payload ổn định, không làm lộ provider detail."""

        request_id = getattr(request, "state", None)
        correlation_id = getattr(request_id, "request_id", str(uuid4()))
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.public_message, "requestId": correlation_id},
            headers={**error.headers, "x-request-id": correlation_id},
        )

    @application.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Gan correlation ID de theo doi Gateway -> AI -> Worker ma khong log payload nhay cam."""

        request.state.request_id = request.headers.get("x-request-id", str(uuid4()))[:128]
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    # Health check chỉ kiểm tra process sống, không gọi OpenAI để tránh phát sinh chi phí.
    @application.get("/api/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """Trả trạng thái sống của service mà không kiểm tra provider trả phí."""

        return {"status": "ok", "service": "ai-service"}

    application.include_router(product_content_router)
    application.include_router(image_optimization_router)
    return application


app = create_app()
