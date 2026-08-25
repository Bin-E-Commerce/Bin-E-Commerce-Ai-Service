"""Adapter OpenAI Responses API, cô lập SDK và structured output khỏi use case."""

from collections.abc import Sequence
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import (
    ConfigurationError,
    InvalidProviderResponseError,
    ProviderUnavailableError,
)
from app.modules.product_content.domain.models import GeneratedName, ProductContext
from app.modules.product_content.infrastructure.prompt_builder import build_description_prompt, build_prompt


# Schema nội bộ buộc model trả title/reason đúng giới hạn trước khi chuyển sang domain.
class _OpenAISuggestion(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=500)
    recommended: bool = False


# Schema bọc ngoài yêu cầu provider trả đúng ba ứng viên trong một response.
class _OpenAIResponse(BaseModel):
    suggestions: list[_OpenAISuggestion] = Field(min_length=3, max_length=3)


# Schema structured output cho mô tả; giới hạn tại adapter để output sai không đi vào application cache.
class _OpenAIDescriptionResponse(BaseModel):
    """Response nội bộ chỉ chứa một mô tả trong giới hạn schema sản phẩm."""

    description: str = Field(min_length=100, max_length=30_000)


# Concrete adapter duy nhất biết SDK OpenAI; lớp khác chỉ phụ thuộc protocol LLM.
class OpenAINameSuggestionProvider:
    """Adapter OpenAI vision có structured output và không lưu response."""

    # Kiểm tra API key lúc dependency được tạo, không làm app import ngầm gọi network.
    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise ConfigurationError()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.openai_model

    # Gửi tối đa ba ảnh CDN, parse schema, rồi chuyển output thành domain model an toàn.
    # Adapter chỉ nhận context đã loại định danh nội bộ; `store=False` bảo đảm request không được lưu bởi provider.
    # Timeout và max_retries=0 giới hạn thời gian/chi phí; lỗi SDK được map thành lỗi công khai ổn định, không lộ secret.
    async def generate_name_suggestions(self, context: ProductContext) -> Sequence[GeneratedName]:
        """Gửi tối đa ba ảnh CDN và structured output; không gửi asset ID hoặc user ID."""

        prompt_text = build_prompt(context)
        image_inputs = [{"type": "input_image", "image_url": image.public_url, "detail": "auto"} for image in context.images]
        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=cast(
                    Any,
                    [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": prompt_text.system}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text.user},
                                *image_inputs,
                            ],
                        },
                    ],
                ),
                text_format=_OpenAIResponse,
                store=False,
            )
        except OpenAIError as error:
            raise ProviderUnavailableError() from error
        except (TimeoutError, OSError) as error:
            raise ProviderUnavailableError() from error
        except (TypeError, ValueError) as error:
            raise InvalidProviderResponseError() from error

        parsed = response.output_parsed
        if parsed is None:
            raise InvalidProviderResponseError()
        return tuple(
            GeneratedName(title=item.title, reason=item.reason, recommended=item.recommended) for item in parsed.suggestions
        )

    # Gọi cùng Responses API nhưng dùng prompt/schema mô tả; store=False và max_retries=0 giữ an toàn chi phí/dữ liệu.
    async def generate_description(self, context: ProductContext) -> str:
        """Sinh một mô tả tiếng Việt structured output từ tối đa ba ảnh CDN."""

        prompt_text = build_description_prompt(context)
        image_inputs = [{"type": "input_image", "image_url": image.public_url, "detail": "auto"} for image in context.images]
        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=cast(
                    Any,
                    [
                        {"role": "system", "content": [{"type": "input_text", "text": prompt_text.system}]},
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt_text.user}, *image_inputs],
                        },
                    ],
                ),
                text_format=_OpenAIDescriptionResponse,
                store=False,
            )
        except OpenAIError as error:
            raise ProviderUnavailableError() from error
        except (TimeoutError, OSError) as error:
            raise ProviderUnavailableError() from error
        except (TypeError, ValueError) as error:
            raise InvalidProviderResponseError() from error

        parsed = response.output_parsed
        if parsed is None:
            raise InvalidProviderResponseError()
        return parsed.description


# Tên alias giúp các import cũ vẫn hoạt động khi adapter đã hỗ trợ cả tên và mô tả.
OpenAIProductContentProvider = OpenAINameSuggestionProvider
