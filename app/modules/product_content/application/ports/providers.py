"""Hợp đồng provider nhỏ theo capability, không ép adapter hỗ trợ chức năng thừa."""

from collections.abc import Sequence
from typing import Protocol

from app.modules.product_content.domain.models import GeneratedName, ProductContext


# Provider đặt tên chỉ cần sinh danh sách tên, không bị ép cài method mô tả.
class ProductNameProvider(Protocol):
    """Capability sinh tên sản phẩm."""

    # Nhận context đã loại asset/user ID và trả candidate chưa sanitize.
    async def generate_name_suggestions(self, context: ProductContext) -> Sequence[GeneratedName]:
        """Sinh đúng ba tên theo schema provider."""


# Provider mô tả chỉ cần sinh mô tả, cho phép dùng model/vendor khác với đặt tên.
class ProductDescriptionProvider(Protocol):
    """Capability sinh mô tả sản phẩm."""

    # Nhận context đã lọc và trả một mô tả để application kiểm tra deterministic.
    async def generate_description(self, context: ProductContext) -> str:
        """Sinh một bản mô tả hoàn chỉnh."""
