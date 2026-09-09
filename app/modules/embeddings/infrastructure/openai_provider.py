"""Adapter OpenAI Embeddings, cô lập SDK trả phí khỏi Kafka worker."""

from asyncio import sleep

from openai import APITimeoutError, AsyncOpenAI, OpenAIError

from app.core.config import Settings


class EmbeddingProviderUnavailableError(RuntimeError):
    """Lỗi tạm thời sau khi provider đã retry bounded; worker sẽ đưa message vào DLQ."""


class ProductEmbeddingProvider:
    """Tạo vector product với timeout/retry bounded và không ghi raw text/vector vào log."""

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY is required for embedding worker")
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.embedding_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.embedding_model
        self._max_retry_attempts = max(1, settings.embedding_max_retry_attempts)

    # Gọi provider với input text đã bounded; retry chỉ áp dụng lỗi transient trước khi message vào DLQ.
    async def generate(self, text: str) -> tuple[str, list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retry_attempts + 1):
            try:
                response = await self._client.embeddings.create(model=self._model, input=text)
                if not response.data or not response.data[0].embedding:
                    raise ValueError("EMBEDDING_EMPTY")
                return response.model or self._model, list(response.data[0].embedding)
            except (APITimeoutError, OpenAIError, OSError, TimeoutError) as error:
                # Các lỗi xác thực/request không thể tự hồi phục; không retry lãng phí và không giữ offset vô hạn.
                permanent_errors = {
                    "AuthenticationError",
                    "PermissionDeniedError",
                    "BadRequestError",
                    "NotFoundError",
                    "UnprocessableEntityError",
                }
                if error.__class__.__name__ in permanent_errors:
                    raise ValueError("EMBEDDING_PROVIDER_REJECTED") from error
                last_error = error
                if attempt >= self._max_retry_attempts:
                    break
                await sleep(min(2**attempt, 16))
        raise EmbeddingProviderUnavailableError("EMBEDDING_PROVIDER_UNAVAILABLE") from last_error
