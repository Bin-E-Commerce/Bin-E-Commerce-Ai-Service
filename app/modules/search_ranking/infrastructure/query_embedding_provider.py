"""OpenAI adapter tạo vector query đồng bộ với embedding catalog đã lưu trong Qdrant."""

import math

from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError


class OpenAiSearchQueryEmbeddingProvider:
    """Một request embedding không retry để giữ latency và chi phí search bị giới hạn."""

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY_REQUIRED_FOR_SEARCH_EMBEDDING")
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.search_query_embedding_timeout_seconds,
            max_retries=0,
        )
        self.model_version = settings.embedding_model_version
        self.dimensions = settings.embedding_dimensions
        self._model = settings.embedding_model

    # Gọi cùng embedding model và dimension với catalog; lỗi provider thành 503 để caller fallback SQL.
    async def generate(self, query: str) -> list[float]:
        """Tạo một vector hữu hạn, không ghi query hoặc vector vào log."""

        try:
            response = await self._client.embeddings.create(model=self._model, input=query)
        except (OpenAIError, OSError, TimeoutError) as error:
            raise ProviderUnavailableError() from error
        if not response.data or not response.data[0].embedding:
            raise ProviderUnavailableError()
        vector = [float(value) for value in response.data[0].embedding]
        if len(vector) != self.dimensions or not all(math.isfinite(value) for value in vector):
            raise ProviderUnavailableError()
        return vector

    # Đóng client riêng khi FastAPI process dừng để không giữ connection pool bên ngoài lifespan.
    async def aclose(self) -> None:
        """Giải phóng HTTP session của OpenAI SDK."""

        await self._client.close()


# Tắt semantic query embedding khi thiếu key, không làm AI Service fail startup ở memory mode.
def build_search_query_embedding_provider(settings: Settings) -> OpenAiSearchQueryEmbeddingProvider | None:
    """Trả provider khi API key hợp lệ; search reranker vẫn có thể chạy độc lập."""

    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        return None
    return OpenAiSearchQueryEmbeddingProvider(settings)
