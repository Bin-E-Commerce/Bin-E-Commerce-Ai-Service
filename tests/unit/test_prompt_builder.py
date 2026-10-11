"""Kiểm tra prompt marketplace và ranh giới redact trước khi gửi dữ liệu tới LLM."""

from app.modules.product_content.domain.models import ProductContext, ProductImage
from app.modules.product_content.infrastructure.prompts.prompt_builder import build_description_prompt, build_prompt


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


# Prompt mô tả phải ép cấu trúc section và nguyên tắc không bịa facts trước khi provider gọi model.
def test_description_prompt_contains_structured_marketplace_rules() -> None:
    prompt = build_description_prompt(product_context())

    required_headings = (
        "Điểm nổi bật:",
        "Thông tin sản phẩm:",
        "Thông số kỹ thuật:",
        "Hướng dẫn sử dụng và bảo quản:",
        "Nguồn gốc và bảo hành:",
        "Bộ sản phẩm gồm:",
        "Lưu ý khi sử dụng:",
    )

    assert all(heading in prompt.system for heading in required_headings)
    assert "Never omit a heading." in prompt.system
    assert "Người bán chưa cung cấp thông tin cho mục này." in prompt.system
    assert 'Do not add "Mô tả chi tiết:"' in prompt.system
    assert "invent specifications" in prompt.system
    assert "550e8400" not in prompt.user
