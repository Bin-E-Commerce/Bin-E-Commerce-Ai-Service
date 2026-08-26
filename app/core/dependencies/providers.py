"""Tạo dependency FastAPI cho config, bảo mật, cache, rate limit và application service."""

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
from app.modules.product_content.application.service import (
    ProductDescriptionSuggestionService,
    ProductNameSuggestionService,
)
from app.modules.product_content.infrastructure.provider_factory import build_llm_provider

SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_image_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Kiem tra permission rieng cua image optimization truoc khi tao job tieu quota."""

    if not settings.ai_image_optimization_enabled:
        raise ConfigurationError()
    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.view")


def get_image_generate_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Kiem tra permission generate rieng de route tao job khong chi dua vao quyen view."""

    if not settings.ai_image_optimization_enabled:
        raise ConfigurationError()
    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.generate")


def get_image_apply_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Kiem tra quyen apply output AI truoc khi goi Product Service thay doi media."""

    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.apply")


def get_image_rollback_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Kiem tra quyen rollback tach rieng de tranh seller chi co view tu y khoi phuc media."""

    return build_user_context(x_user_id, x_user_permissions, "seller.ai.image_optimization.rollback")


async def get_image_optimization_service(request: Request) -> AsyncIterator[ImageOptimizationApplicationService]:
    """Tao service theo request; production dung PostgreSQL+outbox, local fallback memory de dev nhanh."""

    settings = get_settings()
    internal_token_configured = bool(settings.internal_service_token and settings.internal_service_token.get_secret_value())
    owner_client = HttpProductOwnerClient(settings) if internal_token_configured else None
    product_media_client = HttpProductMediaClient(settings) if internal_token_configured else None
    media_asset_client = HttpMediaAssetClient(settings) if internal_token_configured else None
    # Chỉ khởi tạo cipher khi có key; request không dùng mô tả tùy chỉnh vẫn chạy được ở local không có secret.
    cipher_secret = (
        settings.ai_image_background_encryption_key.get_secret_value() if settings.ai_image_background_encryption_key else None
    )
    background_cipher = FernetBackgroundDescriptionCipher(cipher_secret) if cipher_secret else None
    session_factory = getattr(request.app.state, "image_session_factory", None)
    if session_factory is None:
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
        )
        return

    async with session_factory() as session:
        service = ImageOptimizationApplicationService(
            repository=SqlAlchemyImageOptimizationJobRepository(session),
            publisher=SqlAlchemyOptimizationOutboxPublisher(session),
            owner_client=owner_client,
            product_media_client=product_media_client,
            media_asset_client=media_asset_client,
            rate_limiter=request.app.state.rate_limiter,
            rate_limit_requests=settings.ai_image_rate_limit_requests,
            rate_limit_window_seconds=settings.ai_image_rate_limit_window_seconds,
            background_cipher=background_cipher,
        )
        try:
            yield service
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Đọc header do Gateway forward và chặn request trước khi khởi tạo provider trả phí.
def get_current_user(
    settings: SettingsDependency,
    x_user_id: Annotated[str | None, Header(alias="x-user-id")] = None,
    x_user_permissions: Annotated[str | None, Header(alias="x-user-permissions")] = None,
) -> UserContext:
    """Đọc context do Gateway forward và chặn request không có permission."""

    return build_user_context(x_user_id, x_user_permissions, settings.required_permission)


# Lắp provider qua factory để thay OpenAI bằng LLM khác mà route và use case không đổi.
def get_product_name_service(
    request: Request,
    settings: SettingsDependency,
) -> ProductNameSuggestionService:
    """Wiring provider qua interface để thay OpenAI bằng LLM khác không đổi route."""

    cache = request.app.state.result_cache
    rate_limiter = request.app.state.rate_limiter
    provider = build_llm_provider(settings)
    return ProductNameSuggestionService(
        provider=provider,
        cache=cache,
        rate_limiter=rate_limiter,
        settings=settings,
    )


# Wiring use case mô tả qua cùng provider/cache/quota; route không cần biết adapter OpenAI cụ thể.
def get_product_description_service(
    request: Request,
    settings: SettingsDependency,
) -> ProductDescriptionSuggestionService:
    """Tạo application service mô tả với các dependency dùng chung theo process."""

    return ProductDescriptionSuggestionService(
        provider=build_llm_provider(settings),
        cache=request.app.state.result_cache,
        rate_limiter=request.app.state.rate_limiter,
        settings=settings,
    )
