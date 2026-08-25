"""Kiểm tra redaction và warning đối với dữ liệu nhạy cảm trong title/prompt."""

from app.modules.product_content.domain.safety import redact_sensitive_text, sanitize_description, sanitize_title


# Đảm bảo UUID và URL bị che trước khi text seller được gửi ra provider.
def test_redacts_sensitive_values_before_prompt() -> None:
    assert redact_sensitive_text("https://internal.example/item 550e8400-e29b-41d4-a716-446655440000") == "[REDACTED] [REDACTED]"


# Đảm bảo warning không làm lộ lại giá trị nhạy cảm đã bị loại khỏi title.
def test_sanitize_title_returns_warning_without_leaking_value() -> None:
    title, warnings = sanitize_title("Giày da nam https://internal.example/item")

    assert title == "Giày da nam"
    assert warnings[0].code == "SENSITIVE_DATA"
    assert "internal.example" not in warnings[0].message


# Mô tả phải lọc URL/hashtag nhưng vẫn giữ nội dung marketplace để seller xem trước.
def test_sanitize_description_removes_sensitive_tokens() -> None:
    description, warnings = sanitize_description(
        "Điểm nổi bật:\n- Chất liệu bền\n\nMô tả chi tiết:\nSản phẩm dễ dùng mỗi ngày. https://internal.example/item #sale"
    )

    assert "https://" not in description
    assert "#sale" not in description
    assert warnings[0].field == "description"
