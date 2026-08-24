"""Kiểm tra use case gợi ý tên bằng fake provider, không gọi OpenAI thật."""

import pytest

from app.core.config import Settings
from app.core.errors import InvalidProviderResponseError
from app.modules.product_content.application.commands import ImageCommand, NameSuggestionCommand
from app.modules.product_content.application.service import ProductNameSuggestionService
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
    service = ProductNameSuggestionService(provider, MemoryResultCache(), MemoryRateLimiter(), Settings())

    _, first = await service.generate(command(), "seller-1")
    _, second = await service.generate(command(), "seller-1")

    assert len(first.suggestions) == 3
    assert first == second


# Provider lỗi dùng để chứng minh service không chấp nhận output thiếu candidate.
class InvalidProvider:
    # Trả shape sai hợp đồng để use case phải map thành lỗi an toàn.
    async def generate_name_suggestions(self, context):
        return (GeneratedName("Tên quá ngắn", "reason", True),)


@pytest.mark.asyncio
# Output sai schema phải bị chặn trước khi trả response hoặc ghi cache.
async def test_service_rejects_invalid_provider_shape() -> None:
    service = ProductNameSuggestionService(InvalidProvider(), MemoryResultCache(), MemoryRateLimiter(), Settings())

    with pytest.raises(InvalidProviderResponseError):
        await service.generate(command(), "seller-1")
