"""Kiểm tra prompt marketplace và ranh giới redact trước khi gửi dữ liệu tới LLM."""

from app.modules.product_content.domain.models import ProductContext, ProductImage
from app.modules.product_content.infrastructure.prompt_builder import build_prompt


# Tạo context tối thiểu nhưng có thuộc tính kỹ thuật để kiểm tra prompt giữ đúng công thức ngành hàng.
def product_context() -> ProductContext:
    return ProductContext(
        category_name="Giày dép",
        category_path="Thời trang > Giày dép",
        brand="Bin Shoes",
        draft_name="Giày sneaker nam",
        short_description="Giày da đế êm",
        description=None,
        attributes=(("Đối tượng", "Nam"), ("Màu", "Đen")),
        images=(ProductImage("https://cdn.example.com/shoe.jpg", "shoe.jpg"),),
        locale="vi-VN",
    )


# Prompt phải chứa công thức marketplace, quy tắc không bịa và yêu cầu đúng ba đề xuất.
def test_prompt_contains_marketplace_naming_rules() -> None:
    prompt = build_prompt(product_context())

    assert "Footwear: [Shoe type] + [Brand] + [Model] + [Target audience] + [Color]." in prompt.system
    assert "Never invent RAM, ROM, CPU, SSD, GPU" in prompt.system
    assert "exactly three suggestions" in prompt.system
    assert "Category: Giày dép" in prompt.user
