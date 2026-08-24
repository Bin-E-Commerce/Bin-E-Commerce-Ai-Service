"""Chọn LLM adapter ở composition boundary, không để route biết provider cụ thể."""

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.modules.product_content.domain.ports import LLMNameSuggestionProvider
from app.modules.product_content.infrastructure.openai_provider import OpenAINameSuggestionProvider


# Đọc LLM_PROVIDER để sau này thêm Anthropic/Gemini/local model mà không sửa use case.
def build_llm_provider(settings: Settings) -> LLMNameSuggestionProvider:
    """Chọn adapter bằng config để thêm LLM mới mà không sửa use case hoặc route."""

    provider_name = settings.llm_provider.strip().lower()
    if provider_name == "openai":
        return OpenAINameSuggestionProvider(settings)
    raise ConfigurationError()
