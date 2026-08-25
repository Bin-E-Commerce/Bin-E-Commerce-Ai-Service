"""Mô hình domain dùng chung giữa use case và các LLM provider có thể thay thế."""

from dataclasses import dataclass


# Chỉ giữ URL CDN và tên file cần cho vision; tuyệt đối không đưa asset ID vào prompt.
@dataclass(frozen=True)
class ProductImage:
    """Thông tin ảnh đã được boundary kiểm tra và có thể gửi cho vision model."""

    public_url: str
    file_name: str


# Gom context nghiệp vụ thành object thuần Python để domain không phụ thuộc Pydantic/FastAPI.
@dataclass(frozen=True)
class ProductContext:
    """Thông tin seller cung cấp, đã loại các định danh nội bộ khỏi provider context."""

    category_name: str
    category_path: str | None
    brand: str | None
    draft_name: str | None
    short_description: str | None
    description: str | None
    attributes: tuple[tuple[str, str], ...]
    images: tuple[ProductImage, ...]
    locale: str


# Đại diện một ứng viên tên sản phẩm do LLM sinh ra trước bước safety validation.
@dataclass(frozen=True)
class GeneratedName:
    """Một tên sản phẩm ứng viên kèm lý do và cờ đề xuất tốt nhất."""

    title: str
    reason: str
    recommended: bool


# Warning chỉ chứa mã và thông báo an toàn, không chứa lại giá trị nhạy cảm bị phát hiện.
@dataclass(frozen=True)
class SafetyWarning:
    """Cảnh báo được tạo bởi validator deterministic sau khi kiểm tra output."""

    code: str
    field: str
    message: str


# Batch immutable để cache không bị thay đổi sau khi đã validate và sanitize.
@dataclass(frozen=True)
class SuggestionBatch:
    """Ba đề xuất và warning đã hợp lệ, đủ an toàn để lưu cache ngắn hạn."""

    suggestions: tuple[GeneratedName, ...]
    warnings: tuple[SafetyWarning, ...]


# Đại diện cho một bản mô tả hoàn chỉnh đã được provider sinh và safety validator kiểm tra.
# Mô tả giữ nguyên cấu trúc section/bullet để frontend chỉ cần hiển thị và seller xác nhận.
@dataclass(frozen=True)
class DescriptionBatch:
    """Một mô tả sản phẩm an toàn kèm cảnh báo đã được chuẩn hóa."""

    description: str
    warnings: tuple[SafetyWarning, ...]
