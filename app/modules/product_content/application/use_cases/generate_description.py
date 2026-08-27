"""Use case sinh một mô tả sản phẩm có cache-before-rate-limit và safety validation."""

from app.core.config import Settings
from app.core.errors import InvalidInputError, InvalidProviderResponseError
from app.modules.product_content.application.commands import DescriptionSuggestionCommand
from app.modules.product_content.application.policies import validate_cdn_images
from app.modules.product_content.application.ports import ProductDescriptionProvider, RateLimiter, ResultCache
from app.modules.product_content.domain.models import DescriptionBatch
from app.modules.product_content.domain.safety import sanitize_description


# Mỗi instance chỉ điều phối capability mô tả và có một public method execute.
class GenerateProductDescription:
    """Validate, cache, rate-limit, gọi provider và sanitize mô tả."""

    # Nhận port riêng cho mô tả nên provider tên không phải cài method này.
    def __init__(
        self, provider: ProductDescriptionProvider, cache: ResultCache, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        """Không tự khởi tạo OpenAI hoặc Redis trong use case."""

        self._provider = provider
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._settings = settings

    # Thứ tự bắt buộc giữ cache hit miễn quota và không phát sinh lời gọi trả phí.
    async def execute(self, command: DescriptionSuggestionCommand, user_id: str) -> DescriptionBatch:
        """Trả duy nhất mô tả đã vượt safety validator."""

        self._validate_limits(command)
        validate_cdn_images(command.images, self._settings.media_public_cdn_url)
        cache_key = command.cache_key()
        cached = await self._cache.get(cache_key)
        if isinstance(cached, DescriptionBatch):
            return cached
        await self._rate_limiter.check(
            key=f"ai:product-content:{user_id}",
            limit=self._settings.ai_rate_limit_requests,
            window_seconds=self._settings.ai_rate_limit_window_seconds,
        )
        description, warnings = sanitize_description(await self._provider.generate_description(command.to_provider_context()))
        if not 100 <= len(description) <= 30_000:
            raise InvalidProviderResponseError()
        result = DescriptionBatch(description=description, warnings=warnings)
        await self._cache.set(cache_key, result, self._settings.ai_cache_ttl_seconds)
        return result

    # Chặn thiếu ảnh, quá số ảnh hoặc text quá lớn trước mọi external dependency.
    def _validate_limits(self, command: DescriptionSuggestionCommand) -> None:
        """Không log raw text khi validation thất bại."""

        values = [
            command.category_name,
            command.category_path,
            command.brand,
            command.draft_name,
            command.description,
            *(value for pair in command.attributes for value in pair),
            *(image.file_name for image in command.images),
        ]
        if (
            not 1 <= len(command.images) <= self._settings.ai_max_images
            or sum(len(value or "") for value in values) > self._settings.ai_max_text_chars
        ):
            raise InvalidInputError()
