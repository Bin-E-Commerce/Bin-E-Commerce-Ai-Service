"""Tạo dependency FastAPI cho config, bảo mật, cache, rate limit và application service."""

from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.security import UserContext, build_user_context
from app.modules.product_content.application.service import (
    ProductDescriptionSuggestionService,
    ProductNameSuggestionService,
)
from app.modules.product_content.infrastructure.provider_factory import build_llm_provider

SettingsDependency = Annotated[Settings, Depends(get_settings)]


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
