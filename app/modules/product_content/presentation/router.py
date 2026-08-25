"""File này định nghĩa router cho module product_content,
giúp mapping HTTP request/response sang command/result của application layer."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, status

from app.core.config import Settings, get_settings
from app.core.dependencies import get_current_user, get_product_description_service, get_product_name_service
from app.core.errors import InvalidInputError
from app.core.security import UserContext
from app.modules.product_content.application.commands import (
    DescriptionSuggestionCommand,
    ImageCommand,
    NameSuggestionCommand,
)
from app.modules.product_content.application.service import (
    ProductDescriptionSuggestionService,
    ProductNameSuggestionService,
)
from app.modules.product_content.presentation.schemas import (
    DescriptionSuggestionRequest,
    DescriptionSuggestionResponse,
    NameSuggestionRequest,
    NameSuggestionResponse,
    SuggestionResponse,
    WarningResponse,
)

router = APIRouter(prefix="/api/v1/seller/product-content", tags=["seller-product-content"])


# Route kiểm tra quyền ở dependency, kiểm tra CDN ở boundary rồi mới gọi application service.
# Mapping response chỉ trả title/reason/warning đã sanitize, không đưa prompt, user context hoặc provider detail ra ngoài.
@router.post(
    "/name-suggestions",
    response_model=NameSuggestionResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    responses={
        403: {"description": "Missing seller AI permission"},
        429: {"description": "Seller rate limit exceeded"},
        502: {"description": "Invalid provider response"},
        503: {"description": "Provider unavailable or not configured"},
    },
)
async def suggest_product_names(
    payload: NameSuggestionRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[ProductNameSuggestionService, Depends(get_product_name_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NameSuggestionResponse:
    """Kiểm tra CDN, chuyển schema thành command, gọi use case và map result an toàn ra JSON."""

    _validate_cdn_hosts(payload, settings)
    command = _to_command(payload)
    request_id, result = await service.generate(command, user.user_id)
    return NameSuggestionResponse(
        suggestions=[
            SuggestionResponse(
                id=f"suggestion-{uuid4()}",
                title=item.title,
                reason=item.reason,
                recommended=item.recommended,
            )
            for item in result.suggestions
        ],
        warnings=[
            WarningResponse(code=warning.code, field=warning.field, message=warning.message) for warning in result.warnings
        ],
        request_id=request_id,
    )


# Endpoint mô tả dùng cùng permission/context/CDN guard nhưng trả một bản preview duy nhất.
@router.post(
    "/description-suggestions",
    response_model=DescriptionSuggestionResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    responses={
        403: {"description": "Missing seller AI permission"},
        429: {"description": "Seller rate limit exceeded"},
        502: {"description": "Invalid provider response"},
        503: {"description": "Provider unavailable or not configured"},
    },
)
async def suggest_product_description(
    payload: DescriptionSuggestionRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[ProductDescriptionSuggestionService, Depends(get_product_description_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DescriptionSuggestionResponse:
    """Validate CDN, gọi use case mô tả và map kết quả an toàn ra HTTP response."""

    _validate_cdn_hosts(payload, settings)
    command = _to_description_command(payload)
    request_id, result = await service.generate(command, user.user_id)
    return DescriptionSuggestionResponse(
        description=result.description,
        warnings=[WarningResponse(code=item.code, field=item.field, message=item.message) for item in result.warnings],
        requestId=request_id,
    )


# Chuyển alias JSON thành command thuần Python để application không phụ thuộc FastAPI.
# Conversion giữ asset ID ở application command phục vụ fingerprint cache, nhưng ProductContext sẽ loại nó trước LLM.
# Nhờ vậy router chỉ làm nhiệm vụ mapping boundary, không chứa prompt, business branching hoặc provider logic.
def _to_command(payload: NameSuggestionRequest) -> NameSuggestionCommand:
    """Chuyển dữ liệu Pydantic ở HTTP boundary thành command độc lập framework cho application layer."""

    seller_input = payload.seller_input
    return NameSuggestionCommand(
        category_name=payload.category.name,
        category_path=payload.category.path,
        brand=payload.brand,
        draft_name=seller_input.draft_name if seller_input else None,
        short_description=seller_input.short_description if seller_input else None,
        description=seller_input.description if seller_input else None,
        attributes=(tuple((attribute.label, attribute.value) for attribute in seller_input.attributes) if seller_input else ()),
        images=tuple(
            ImageCommand(
                asset_id=image.asset_id,
                public_url=str(image.public_url),
                file_name=image.file_name,
            )
            for image in payload.images
        ),
        locale=payload.locale,
    )


# Chuyển request mô tả thành command thuần Python; shortDescription không được đưa vào use case này.
def _to_description_command(payload: DescriptionSuggestionRequest) -> DescriptionSuggestionCommand:
    """Map schema mô tả sang application command, giữ asset ID chỉ cho fingerprint cache."""

    seller_input = payload.seller_input
    return DescriptionSuggestionCommand(
        category_name=payload.category.name,
        category_path=payload.category.path,
        brand=payload.brand,
        draft_name=seller_input.draft_name if seller_input else None,
        description=seller_input.description if seller_input else None,
        attributes=(tuple((item.label, item.value) for item in seller_input.attributes) if seller_input else ()),
        images=tuple(
            ImageCommand(asset_id=image.asset_id, public_url=str(image.public_url), file_name=image.file_name)
            for image in payload.images
        ),
        locale=payload.locale,
    )


# Chỉ cho phép HTTPS origin đã cấu hình để tránh SSRF và không gửi URL nội bộ cho provider.
# Origin được chuẩn hóa từ scheme + host + port, sau đó so khớp chính xác với allow-list cấu hình.
# Không dùng kiểm tra chuỗi tiền tố vì `cdn.example.com.attacker.test` không được phép giả dạng CDN hợp lệ.
def _validate_cdn_hosts(payload: NameSuggestionRequest | DescriptionSuggestionRequest, settings: Settings) -> None:
    """Chỉ cho phép HTTPS và origin CDN đã cấu hình, không chấp nhận URL nội bộ."""

    from urllib.parse import urlparse

    allowed_origins = {origin.strip().rstrip("/") for origin in settings.media_public_cdn_url.split(",") if origin.strip()}
    if not allowed_origins:
        raise InvalidInputError()

    for image in payload.images:
        parsed = urlparse(str(image.public_url))
        origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        if parsed.scheme != "https" or origin not in allowed_origins:
            raise InvalidInputError()
