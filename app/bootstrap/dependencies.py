"""FastAPI dependency wiring cho application use cases và infrastructure adapters.

Business module không tự đọc settings hoặc khởi tạo HTTP/database client. Memory
fallback chỉ được dùng khi lifespan đã xác nhận runtime mode `memory`.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.core.security import UserContext, build_user_context
from app.modules.image_optimization.application.service import ImageOptimizationApplicationService
from app.modules.image_optimization.infrastructure.clients import (
    HttpMediaAssetClient,
    HttpProductMediaClient,
    HttpProductOwnerClient,
)
from app.modules.image_optimization.infrastructure.persistence.outbox_publisher import SqlAlchemyOptimizationOutboxPublisher
from app.modules.image_optimization.infrastructure.persistence.sqlalchemy_repository import (
    SqlAlchemyImageOptimizationJobRepository,
)
from app.modules.image_optimization.infrastructure.security import FernetBackgroundDescriptionCipher
from app.modules.product_content.application.use_cases import GenerateProductDescription, GenerateProductNames

SettingsDependency = Annotated[Settings, Depends(get_settings)]


# Xác thực permission view trước khi đọc job/overview của seller.
def get_image_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_email: Annotated[str | None, Header(alias="x-user-email")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Chặn feature disabled và context thiếu quyền ở HTTP boundary."""

    if not settings.ai_image_optimization_enabled:
        raise ConfigurationError()
    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.view", x_user_email)


# Xác thực permission generate trước khi request có thể tiêu quota hoặc tạo outbox.
def get_image_generate_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_email: Annotated[str | None, Header(alias="x-user-email")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Trả immutable UserContext đã parse permission một lần."""

    if not settings.ai_image_optimization_enabled:
        raise ConfigurationError()
    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.generate", x_user_email)


# Xác thực permission apply trước khi Product Service mutation được gọi.
def get_image_apply_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_email: Annotated[str | None, Header(alias="x-user-email")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Giữ nguyên toàn bộ permission đã xác thực để downstream không phải tự tạo policy."""

    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.apply", x_user_email)


# Xác thực quyền rollback riêng với quyền apply.
def get_image_rollback_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_email: Annotated[str | None, Header(alias="x-user-email")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Không cho user chỉ có view tự khôi phục media sản phẩm."""

    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.rollback", x_user_email)


# Wiring facade từ từng adapter theo đúng runtime mode đã được lifespan kiểm tra.
async def get_image_optimization_service(request: Request) -> AsyncIterator[ImageOptimizationApplicationService]:
    """Production dùng PostgreSQL/outbox/shared HTTP; memory mode dùng explicit in-process adapters."""

    settings = get_settings()
    memory_mode = request.app.state.ai_runtime_mode == "memory"
    shared_http_client = request.app.state.http_client
    owner_client = None if memory_mode else HttpProductOwnerClient(settings, shared_http_client)
    product_media_client = None if memory_mode else HttpProductMediaClient(settings, shared_http_client)
    media_asset_client = None if memory_mode else HttpMediaAssetClient(settings, shared_http_client)
    cipher_secret = (
        settings.ai_image_background_encryption_key.get_secret_value() if settings.ai_image_background_encryption_key else None
    )
    background_cipher = FernetBackgroundDescriptionCipher(cipher_secret) if cipher_secret else None

    if memory_mode:
        yield ImageOptimizationApplicationService(
            repository=request.app.state.image_optimization_repository,
            publisher=request.app.state.image_optimization_publisher,
            owner_client=owner_client,
            product_media_client=product_media_client,
            media_asset_client=media_asset_client,
            rate_limiter=request.app.state.rate_limiter,
            rate_limit_requests=settings.ai_image_rate_limit_requests,
            rate_limit_window_seconds=settings.ai_image_rate_limit_window_seconds,
            background_cipher=background_cipher,
            allow_memory_adapters=True,
        )
        return

    session_factory = request.app.state.image_session_factory
    async with session_factory() as session, session.begin():
        yield ImageOptimizationApplicationService(
            repository=SqlAlchemyImageOptimizationJobRepository(session),
            publisher=SqlAlchemyOptimizationOutboxPublisher(session),
            owner_client=owner_client,
            product_media_client=product_media_client,
            media_asset_client=media_asset_client,
            rate_limiter=request.app.state.rate_limiter,
            rate_limit_requests=settings.ai_image_rate_limit_requests,
            rate_limit_window_seconds=settings.ai_image_rate_limit_window_seconds,
            background_cipher=background_cipher,
            allow_memory_adapters=False,
            finalize_before_apply=True,
        )


# Đọc identity/permission do Gateway chuyển tiếp cho product-content endpoints.
def get_current_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_email: Annotated[str | None, Header(alias="x-user-email")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Chặn request thiếu quyền trước khi khởi tạo provider trả phí."""

    return build_user_context(x_user_id, x_user_permissions, settings.required_permission, x_user_email)


# Wiring use case gợi ý tên với cache/rate limiter đã được lifespan chọn.
def get_product_name_service(request: Request, settings: SettingsDependency) -> GenerateProductNames:
    """Provider registry được tạo ở composition root, không trong router/use case."""

    return GenerateProductNames(
        provider=request.app.state.product_name_provider,
        cache=request.app.state.result_cache,
        rate_limiter=request.app.state.rate_limiter,
        settings=settings,
    )


# Wiring use case mô tả với cùng shared cache/quota nhưng capability độc lập.
def get_product_description_service(request: Request, settings: SettingsDependency) -> GenerateProductDescription:
    """Không tạo mutable global provider client khi import module."""

    return GenerateProductDescription(
        provider=request.app.state.product_description_provider,
        cache=request.app.state.result_cache,
        rate_limiter=request.app.state.rate_limiter,
        settings=settings,
    )
