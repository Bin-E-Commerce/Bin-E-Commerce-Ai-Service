"""Use case sinh ba tên sản phẩm, có cache-before-rate-limit và safety validation."""

from collections.abc import Sequence

from app.core.config import Settings
from app.core.errors import InvalidInputError, InvalidProviderResponseError
from app.modules.product_content.application.commands import NameSuggestionCommand
from app.modules.product_content.application.policies import validate_cdn_images
from app.modules.product_content.application.ports import ProductNameProvider, RateLimiter, ResultCache
from app.modules.product_content.domain.models import GeneratedName, SafetyWarning, SuggestionBatch
from app.modules.product_content.domain.safety import redact_sensitive_text, sanitize_title


# Mỗi instance điều phối đúng một capability và chỉ có một public method execute.
class GenerateProductNames:
    """Validate, cache, rate-limit, gọi provider và sanitize ba tên đề xuất."""

    # Nhận dependency qua port để unit test không gọi OpenAI/Redis thật.
    def __init__(self, provider: ProductNameProvider, cache: ResultCache, rate_limiter: RateLimiter, settings: Settings) -> None:
        """Lưu policy runtime nhưng không tự đọc env hoặc tạo network client."""

        self._provider = provider
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._settings = settings

    # Thứ tự bắt buộc: validate → cache → rate limit → paid provider → safety → cache.
    async def execute(self, command: NameSuggestionCommand, user_id: str) -> SuggestionBatch:
        """Cache hit không tiêu thêm quota hoặc gọi provider trả phí."""

        self._validate_limits(command)
        validate_cdn_images(command.images, self._settings.media_public_cdn_url)
        cache_key = command.cache_key()
        cached = await self._cache.get(cache_key)
        if isinstance(cached, SuggestionBatch):
            return cached
        await self._rate_limiter.check(
            key=f"ai:product-content:{user_id}",
            limit=self._settings.ai_rate_limit_requests,
            window_seconds=self._settings.ai_rate_limit_window_seconds,
        )
        result = self._sanitize(await self._provider.generate_name_suggestions(command.to_provider_context()))
        await self._cache.set(cache_key, result, self._settings.ai_cache_ttl_seconds)
        return result

    # Chặn payload lớn trước cache/provider để bảo vệ memory và context window.
    def _validate_limits(self, command: NameSuggestionCommand) -> None:
        """Kiểm tra số ảnh và tổng text mà không ghi nội dung vào log."""

        values = [
            command.category_name,
            command.category_path,
            command.brand,
            command.draft_name,
            command.short_description,
            command.description,
            *(value for pair in command.attributes for value in pair),
            *(image.file_name for image in command.images),
        ]
        if (
            len(command.images) > self._settings.ai_max_images
            or sum(len(value or "") for value in values) > self._settings.ai_max_text_chars
        ):
            raise InvalidInputError()

    # Ép exact-three, unique title và đúng một recommended sau deterministic safety.
    def _sanitize(self, generated: Sequence[GeneratedName]) -> SuggestionBatch:
        """Từ chối cả batch khi provider trả shape nửa hợp lệ."""

        if len(generated) != 3:
            raise InvalidProviderResponseError()
        suggestions: list[GeneratedName] = []
        warnings: list[SafetyWarning] = []
        seen: set[str] = set()
        for candidate in generated:
            title, title_warnings = sanitize_title(candidate.title)
            normalized = title.casefold()
            if not 20 <= len(title) <= 200 or normalized in seen:
                raise InvalidProviderResponseError()
            seen.add(normalized)
            suggestions.append(
                GeneratedName(
                    title, (redact_sensitive_text(candidate.reason) or "Generated from verified product context.")[:400], False
                )
            )
            warnings.extend(title_warnings)
        recommended_index = next((index for index, value in enumerate(generated) if value.recommended), 0)
        normalized_suggestions = tuple(
            GeneratedName(value.title, value.reason, index == recommended_index) for index, value in enumerate(suggestions)
        )
        unique_warnings = tuple({(value.code, value.field): value for value in warnings}.values())
        return SuggestionBatch(suggestions=normalized_suggestions, warnings=unique_warnings)
