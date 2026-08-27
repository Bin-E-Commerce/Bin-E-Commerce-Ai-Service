"""Kiểm tra use case gợi ý tên bằng fake provider, không gọi OpenAI thật."""

import pytest

from app.core.config import Settings
from app.core.errors import InvalidProviderResponseError
from app.modules.product_content.application.commands import (
    DescriptionSuggestionCommand,
    ImageCommand,
    NameSuggestionCommand,
)
from app.modules.product_content.application.use_cases import GenerateProductDescription, GenerateProductNames
from app.modules.product_content.domain.models import GeneratedName
from app.modules.product_content.infrastructure.memory_cache import MemoryResultCache
from app.modules.product_content.infrastructure.memory_rate_limiter import MemoryRateLimiter


# Fake provider trả output hợp lệ để test cache và normalize mà không phát sinh chi phí.
class FakeProvider:
    # Tạo đúng ba candidate giống hợp đồng structured output của provider.
    async def generate_name_suggestions(self, context):
        return (
            GeneratedName(
                "Giày da nam công sở cao cấp đế êm Việt Nam",
                "Nêu chất liệu và nhóm khách hàng.",
                True,
            ),
            GeneratedName(
                "Giày nam da mềm phong cách lịch lãm đi làm",
                "Tập trung vào phong cách sử dụng.",
                False,
            ),
            GeneratedName(
                "Giày da nam thiết kế thanh lịch sử dụng hằng ngày",
                "Mô tả kiểu dáng và nhu cầu.",
                False,
            ),
        )


# Rate limiter fake đếm số lần để chứng minh cache hit không tiêu quota lần hai.
class CountingRateLimiter:
    """Không chặn request; chỉ ghi số lần use case thật sự kiểm tra quota."""

    # Khởi tạo counter trước lần execute đầu.
    def __init__(self) -> None:
        """Counter bắt đầu từ zero."""

        self.calls = 0

    # Tăng counter và bỏ các tham số không cần cho assertion.
    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Mô phỏng adapter quota thành công."""

        del key, limit, window_seconds
        self.calls += 1


# Tạo command ổn định để các test tập trung vào từng behavior của application service.
def command() -> NameSuggestionCommand:
    return NameSuggestionCommand(
        category_name="Giày dép",
        category_path=None,
        brand=None,
        draft_name=None,
        short_description=None,
        description=None,
        attributes=(),
        images=(ImageCommand("asset-1", "https://cdn.example.com/a.jpg", "a.jpg"),),
        locale="vi-VN",
    )


@pytest.mark.asyncio
# Hai lần gọi cùng input phải cho cùng batch và lần hai lấy từ cache.
async def test_service_returns_three_suggestions_and_caches_result() -> None:
    provider = FakeProvider()
    rate_limiter = CountingRateLimiter()
    service = GenerateProductNames(
        provider,
        MemoryResultCache(),
        rate_limiter,
        Settings(media_public_cdn_url="https://cdn.example.com"),
    )

    first = await service.execute(command(), "seller-1")
    second = await service.execute(command(), "seller-1")

    assert len(first.suggestions) == 3
    assert first == second
    assert rate_limiter.calls == 1


# Provider lỗi dùng để chứng minh service không chấp nhận output thiếu candidate.
class InvalidProvider:
    # Trả shape sai hợp đồng để use case phải map thành lỗi an toàn.
    async def generate_name_suggestions(self, context):
        return (GeneratedName("Tên quá ngắn", "reason", True),)


@pytest.mark.asyncio
# Output sai schema phải bị chặn trước khi trả response hoặc ghi cache.
async def test_service_rejects_invalid_provider_shape() -> None:
    service = GenerateProductNames(
        InvalidProvider(),
        MemoryResultCache(),
        MemoryRateLimiter(),
        Settings(media_public_cdn_url="https://cdn.example.com"),
    )

    with pytest.raises(InvalidProviderResponseError):
        await service.execute(command(), "seller-1")


# Fake provider mô phỏng nội dung đủ dài nhưng chứa URL để kiểm tra lớp safety sau provider.
class DescriptionProvider:
    # Trả mô tả có section marketplace và token nhạy cảm để service phải loại bỏ URL trước khi cache.
    async def generate_description(self, context):
        return "Điểm nổi bật:\n- Chất liệu bền\n- Thiết kế dễ dùng\n\nMô tả chi tiết:\nSản phẩm phù hợp sử dụng hằng ngày. https://internal.example/item"


# Tạo command mô tả tối thiểu với ảnh CDN hợp lệ cho các test use case.
def description_command() -> DescriptionSuggestionCommand:
    return DescriptionSuggestionCommand(
        category_name="Giày dép",
        category_path=None,
        brand="Bin",
        draft_name="Giày da nam công sở",
        description=None,
        attributes=(("Chất liệu", "Da"),),
        images=(ImageCommand("asset-description-1", "https://cdn.example.com/shoe.jpg", "shoe.jpg"),),
        locale="vi-VN",
    )


@pytest.mark.asyncio
# Hai lần gọi mô tả cùng input phải dùng batch đã sanitize trong cache và không gọi provider lần hai.
async def test_description_service_sanitizes_and_caches_result() -> None:
    provider = DescriptionProvider()
    service = GenerateProductDescription(
        provider,
        MemoryResultCache(),
        MemoryRateLimiter(),
        Settings(media_public_cdn_url="https://cdn.example.com"),
    )

    first = await service.execute(description_command(), "seller-description")
    second = await service.execute(description_command(), "seller-description")

    assert "https://" not in first.description
    assert len(first.description) >= 100
    assert first == second
