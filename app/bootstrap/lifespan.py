"""Khởi tạo và đóng tài nguyên dùng chung theo vòng đời FastAPI application.

Production fail-fast khi thiếu PostgreSQL, Redis, service token hoặc OpenAI key;
memory adapter chỉ hoạt động khi `AI_RUNTIME_MODE=memory` được cấu hình rõ.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.modules.image_optimization.infrastructure.publisher import InMemoryOptimizationEventPublisher
from app.modules.image_optimization.infrastructure.repository import InMemoryImageOptimizationJobRepository
from app.modules.product_content.infrastructure.memory_cache import MemoryResultCache
from app.modules.product_content.infrastructure.memory_rate_limiter import MemoryRateLimiter
from app.modules.product_content.infrastructure.provider_factory import build_product_content_providers
from app.shared.infrastructure.redis import RedisRateLimiter, RedisResultCache


# Kiểm tra cấu hình trước khi mở socket để production không chạy ở trạng thái fail-open.
def validate_runtime_settings(settings: Settings) -> None:
    """Raise lỗi không chứa secret khi runtime service thiếu dependency bắt buộc."""

    if settings.node_env == "production" and settings.ai_runtime_mode != "service":
        raise RuntimeError("Production AI Service requires AI_RUNTIME_MODE=service")
    if settings.ai_runtime_mode == "memory":
        return
    missing: list[str] = []
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if not settings.redis_url:
        missing.append("REDIS_URL")
    if not settings.internal_service_token or not settings.internal_service_token.get_secret_value():
        missing.append("INTERNAL_SERVICE_TOKEN")
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
        missing.append("OPENAI_API_KEY")
    if settings.ai_image_optimization_enabled and not settings.ai_image_background_encryption_key:
        missing.append("AI_IMAGE_BACKGROUND_ENCRYPTION_KEY")
    if missing:
        raise RuntimeError(f"AI Service runtime configuration is incomplete: {', '.join(missing)}")


# Mở connection pool một lần và gắn adapter phù hợp vào application state.
@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Đóng HTTP, Redis và database pool khi process shutdown."""

    configure_logging()
    settings = get_settings()
    validate_runtime_settings(settings)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    application.state.http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(30.0, connect=5.0))
    application.state.ai_runtime_mode = settings.ai_runtime_mode
    name_provider, description_provider = build_product_content_providers(settings)
    application.state.product_name_provider = name_provider
    application.state.product_description_provider = description_provider

    if settings.ai_runtime_mode == "memory":
        application.state.result_cache = MemoryResultCache()
        application.state.rate_limiter = MemoryRateLimiter()
        application.state.image_optimization_repository = InMemoryImageOptimizationJobRepository()
        application.state.image_optimization_publisher = InMemoryOptimizationEventPublisher()
    else:
        from redis.asyncio import Redis
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        if settings.database_url is None or settings.redis_url is None:
            raise RuntimeError("Validated service runtime is missing database or Redis configuration")
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        application.state.image_session_factory = async_sessionmaker(engine, expire_on_commit=False)
        application.state.image_database_engine = engine
        application.state.redis_client = redis_client
        application.state.result_cache = RedisResultCache(redis_client)
        application.state.rate_limiter = RedisRateLimiter(redis_client)

    try:
        yield
    finally:
        await application.state.http_client.aclose()
        redis_client = getattr(application.state, "redis_client", None)
        if redis_client is not None:
            await redis_client.aclose()
        stored_engine = getattr(application.state, "image_database_engine", None)
        if stored_engine is not None:
            await stored_engine.dispose()
