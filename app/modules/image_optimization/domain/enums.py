"""Khai bao enum va trang thai vong doi cua job toi uu anh san pham."""

from enum import StrEnum


class ImageOptimizationMode(StrEnum):
    """Cac kieu output duoc seller chon cho mot job."""

    WHITE_BACKGROUND = "WHITE_BACKGROUND"
    LIFESTYLE_BACKGROUND = "LIFESTYLE_BACKGROUND"


class ImageGenerationProfile(StrEnum):
    """Hồ sơ chất lượng dùng để tách bản xem nhanh và ảnh cuối cùng."""

    PREVIEW = "PREVIEW"
    FINAL = "FINAL"


# Xác định các bối cảnh lifestyle đã được biên tập sẵn để prompt nhất quán và không cần seller viết mô tả cho mọi lần tạo ảnh.
class LifestyleBackgroundPreset(StrEnum):
    """Các lựa chọn bối cảnh an toàn mà seller có thể dùng cho ảnh lifestyle."""

    MINIMAL_STUDIO = "MINIMAL_STUDIO"
    WARM_HOME = "WARM_HOME"
    NATURAL_OUTDOOR = "NATURAL_OUTDOOR"
    PREMIUM_DISPLAY = "PREMIUM_DISPLAY"


# Hiển thị tiến trình thật từ worker để frontend không phải suy đoán chỉ từ trạng thái tổng quát của job.
class ImageOptimizationProcessingStage(StrEnum):
    """Các chặng xử lý nội bộ có thể công khai an toàn cho seller."""

    QUEUED = "QUEUED"
    FETCHING_SOURCE = "FETCHING_SOURCE"
    PREPARING_IMAGE = "PREPARING_IMAGE"
    GENERATING = "GENERATING"
    UPLOADING = "UPLOADING"
    READY = "READY"
    FAILED = "FAILED"


class ImageOptimizationStatus(StrEnum):
    """Trang thai job phuc vu retry, review va apply idempotent."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FINALIZING = "FINALIZING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    APPLIED = "APPLIED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
