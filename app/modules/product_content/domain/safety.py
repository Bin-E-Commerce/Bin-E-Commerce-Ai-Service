"""Regex deterministic để che và loại dữ liệu nhạy cảm khỏi product content."""

import re

from app.modules.product_content.domain.models import SafetyWarning

_SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I),
        "UUID",
    ),
    (re.compile(r"https?://\S+|www\.\S+", re.I), "URL"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]+\b", re.I), "API key"),
    (
        re.compile(r"\b(?:assetId|productId|sellerOwnerId)\b\s*[:=]?\s*[^\s,;]+", re.I),
        "internal identifier",
    ),
    (re.compile(r"/(?:api|internal|private)(?:/|\b)\S*", re.I), "internal path"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.I), "email"),
    (re.compile(r"(?<!\d)(?:\+?84|0)(?:\s|[-.]?\d){8,12}(?!\d)"), "phone number"),
)


# Che dữ liệu trước prompt để seller không vô tình gửi UUID, URL nội bộ hoặc secret cho LLM.
# Các regex được áp dụng tuần tự để một giá trị bị che ở bước trước không thể bị gửi nguyên dạng ở bước sau.
# Hàm giữ nguyên hình dạng văn bản ở mức tối thiểu, nhưng không bao giờ trả lại giá trị gốc của dữ liệu nhạy cảm.
def redact_sensitive_text(value: str | None) -> str | None:
    """Che dữ liệu nhạy cảm trước khi đưa text seller vào prompt."""

    if value is None:
        return None
    redacted = value
    for pattern, _ in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


# Làm sạch title sau output, đồng thời tạo warning nhưng không trả lại giá trị nhạy cảm đã xóa.
# Validator phát hiện trước rồi mới xóa từng loại dữ liệu, sau đó chuẩn hóa khoảng trắng và dấu câu thừa.
# Warning chỉ nêu loại vi phạm; không đưa URL, UUID, email hoặc số điện thoại cụ thể vào response.
def sanitize_title(title: str) -> tuple[str, tuple[SafetyWarning, ...]]:
    """Loại dữ liệu nhạy cảm khỏi title và tạo warning không chứa giá trị gốc."""

    sanitized = title.strip()
    detected: list[str] = []
    for pattern, label in _SENSITIVE_PATTERNS:
        if pattern.search(sanitized):
            detected.append(label)
            sanitized = pattern.sub("", sanitized)

    sanitized = re.sub(r"\s+", " ", sanitized).strip(" -–—,;:")
    warnings: tuple[SafetyWarning, ...] = ()
    if detected:
        warnings = (
            SafetyWarning(
                code="SENSITIVE_DATA",
                field="title",
                message="Tên đề xuất đã được loại bỏ thông tin nhận diện nội bộ để an toàn.",
            ),
        )
    return sanitized, warnings
