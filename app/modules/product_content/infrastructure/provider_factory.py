"""Chọn LLM adapter ở composition boundary, không để route biết provider cụ thể."""

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.modules.product_content.application.ports import ProductDescriptionProvider, ProductNameProvider
from app.modules.product_content.infrastructure.openai_provider import OpenAINameSuggestionProvider


# Tạo registry capability một lần ở lifespan; vendor branching không xuất hiện trong route/use case.
def build_product_content_providers(settings: Settings) -> tuple[ProductNameProvider, ProductDescriptionProvider]:
    """Trả provider theo từng capability và fail-fast với cấu hình chưa được hỗ trợ."""

    if settings.product_name_provider != "openai" or settings.product_description_provider != "openai":
        raise ConfigurationError()
    openai_provider = OpenAINameSuggestionProvider(settings)
    return openai_provider, openai_provider


# Alias tạm cho integration cũ; code bootstrap mới dùng registry capability ở trên.
def build_llm_provider(settings: Settings) -> ProductNameProvider | ProductDescriptionProvider:
    """Giữ import cũ một release mà không thay đổi provider selection semantics."""

    return build_product_content_providers(settings)[0]
