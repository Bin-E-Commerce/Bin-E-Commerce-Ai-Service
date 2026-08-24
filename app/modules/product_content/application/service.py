"""File này định nghĩa service cho module product_content,
giúp validate input, giới hạn seller, gọi LLM và sanitize output trước cache."""

from collections.abc import Sequence
from uuid import uuid4

from app.core.config import Settings
from app.core.errors import InvalidInputError, InvalidProviderResponseError
from app.modules.product_content.application.commands import NameSuggestionCommand
from app.modules.product_content.domain.models import GeneratedName, SafetyWarning, SuggestionBatch
from app.modules.product_content.domain.ports import (
    LLMNameSuggestionProvider,
    RateLimiter,
    ResultCache,
)
from app.modules.product_content.domain.safety import redact_sensitive_text, sanitize_title


# Đây là điểm duy nhất điều phối nghiệp vụ, giúp router mỏng và provider có thể thay thế.
class ProductNameSuggestionService:
    """Service validate input, giới hạn seller, gọi LLM và sanitize output trước cache."""

    # Nhận các port thay vì concrete adapter để unit test không gọi mạng hoặc tiêu tiền.
    def __init__(
        self,
        provider: LLMNameSuggestionProvider,
        cache: ResultCache,
        rate_limiter: RateLimiter,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._settings = settings

    # Luồng xử lý: validate giới hạn → rate limit → cache → gọi LLM → sanitize → cache kết quả.
    # Thứ tự này chặn payload quá lớn và caller vượt quota trước khi phát sinh chi phí provider.
    # Chỉ output đã kiểm tra deterministic mới được ghi cache; lỗi provider không được retry ngầm hay lưu lại.
    async def generate(self, command: NameSuggestionCommand, user_id: str) -> tuple[str, SuggestionBatch]:
        """Thực thi use case với giới hạn seller và không retry request trả phí."""

        self._validate_command_limits(command)
        await self._rate_limiter.check(
            key=f"ai:name-suggestions:{user_id}",
            limit=self._settings.ai_rate_limit_requests,
            window_seconds=self._settings.ai_rate_limit_window_seconds,
        )

        # Cache hit trả lại kết quả đã sanitize, tránh gọi LLM lặp và phát sinh chi phí.
        cache_key = command.cache_key()
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return str(uuid4()), cached

        generated = await self._provider.generate_name_suggestions(command.to_provider_context())
        batch = self._validate_and_sanitize(generated)
        await self._cache.set(cache_key, batch, self._settings.ai_cache_ttl_seconds)
        return str(uuid4()), batch

    # Chặn payload quá lớn trước rate limit/provider để bảo vệ chi phí và context window.
    def _validate_command_limits(self, command: NameSuggestionCommand) -> None:
        """Chặn payload quá lớn trước khi tốn quota hoặc gửi dữ liệu ra ngoài."""

        if len(command.images) > self._settings.ai_max_images:
            raise InvalidInputError()
        text_values = [
            command.category_name,
            command.category_path,
            command.brand,
            command.draft_name,
            command.short_description,
            command.description,
            *(value for pair in command.attributes for value in pair),
            *(image.file_name for image in command.images),
        ]
        if sum(len(value or "") for value in text_values) > self._settings.ai_max_text_chars:
            raise InvalidInputError()

    # Kiểm tra exact-three, loại title trùng/ngắn và ép đúng một cờ recommended trước khi trả API.
    # Mỗi title được sanitize độc lập để warning có thể trả về mà không làm lộ giá trị nhạy cảm đã xóa.
    # Nếu provider sai số lượng, độ dài hoặc bị trùng, toàn bộ batch bị từ chối thay vì trả dữ liệu nửa hợp lệ.
    def _validate_and_sanitize(self, generated: Sequence[GeneratedName]) -> SuggestionBatch:
        """Đảm bảo provider trả đúng 3 title hợp lệ, duy nhất và chỉ một đề xuất tốt nhất."""

        if len(generated) != 3:
            raise InvalidProviderResponseError()

        suggestions: list[GeneratedName] = []
        warnings: list[SafetyWarning] = []
        seen_titles: set[str] = set()
        for candidate in generated:
            title, title_warnings = sanitize_title(candidate.title)
            if not 20 <= len(title) <= 200:
                raise InvalidProviderResponseError()
            normalized_title = title.casefold()
            if normalized_title in seen_titles:
                raise InvalidProviderResponseError()
            seen_titles.add(normalized_title)
            reason = redact_sensitive_text(candidate.reason) or "Generated from the supplied product context."
            suggestions.append(GeneratedName(title=title, reason=reason[:400], recommended=False))
            warnings.extend(title_warnings)

        recommended_index = next(
            (index for index, candidate in enumerate(generated) if candidate.recommended),
            0,
        )
        normalized_suggestions = tuple(
            GeneratedName(title=item.title, reason=item.reason, recommended=index == recommended_index)
            for index, item in enumerate(suggestions)
        )
        unique_warnings = tuple({(warning.code, warning.field): warning for warning in warnings}.values())
        return SuggestionBatch(suggestions=normalized_suggestions, warnings=unique_warnings)
