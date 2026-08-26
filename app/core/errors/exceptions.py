"""Định nghĩa các lỗi nghiệp vụ và hạ tầng được trả về qua HTTP một cách an toàn."""


# Lỗi nền tảng để mọi lỗi bên trong có mã, HTTP status và thông báo công khai ổn định.
class AppError(Exception):
    """Lỗi cơ sở không được chứa secret, prompt hoặc chi tiết provider nội bộ."""

    status_code = 500
    code = "AI_SERVICE_ERROR"
    public_message = "The AI service could not complete the request."
    headers: dict[str, str] = {}


# Lỗi này chặn request trước khi gọi LLM nếu Gateway không gửi đủ identity hoặc permission.
class AuthorizationError(AppError):
    """Thiếu context người dùng hoặc quyền AI bắt buộc."""

    status_code = 403
    code = "FORBIDDEN"
    public_message = "The required AI permission is missing."


# Lỗi này bảo vệ quota bằng cách dừng payload sai hoặc vượt giới hạn trước khi gọi provider.
class InvalidInputError(AppError):
    """Request vi phạm giới hạn dữ liệu hoặc quy tắc đầu vào."""

    status_code = 422
    code = "AI_INVALID_INPUT"
    public_message = "The AI request contains invalid or unsupported data."


# Lỗi cấu hình giúp service fail an toàn khi thiếu API key hoặc chọn provider chưa hỗ trợ.
class ConfigurationError(AppError):
    """Provider chưa có cấu hình an toàn để thực hiện request."""

    status_code = 503
    code = "AI_CONFIGURATION_ERROR"
    public_message = "The AI provider is not configured."


# Lỗi riêng cho workflow nền tùy chỉnh khi service chưa có khóa mã hóa dùng chung giữa API và worker.
class BackgroundConfigurationError(AppError):
    """Cấu hình bảo mật của mô tả background tùy chỉnh chưa sẵn sàng."""

    status_code = 503
    code = "AI_BACKGROUND_CONFIGURATION_ERROR"
    public_message = (
        "Custom background generation is not configured. Set AI_IMAGE_BACKGROUND_ENCRYPTION_KEY and restart API and worker."
    )


# Lỗi này che chi tiết mạng/quota của provider để không làm lộ thông tin hệ thống ra API.
class ProviderUnavailableError(AppError):
    """LLM provider timeout, lỗi mạng, hết quota hoặc tạm thời không khả dụng."""

    status_code = 503
    code = "AI_PROVIDER_UNAVAILABLE"
    public_message = "The AI provider is temporarily unavailable."


# Lỗi này bảo đảm output không hợp schema không được ghi vào cache hoặc trả cho frontend.
class InvalidProviderResponseError(AppError):
    """Provider trả structured output không đúng hợp đồng nghiệp vụ."""

    status_code = 502
    code = "AI_INVALID_PROVIDER_RESPONSE"
    public_message = "The AI provider returned an unusable response."


# Lỗi này giới hạn chi phí theo seller trong một cửa sổ thời gian cố định.
class RateLimitExceededError(AppError):
    """Seller đã vượt số request AI được phép trong cửa sổ hiện tại."""

    status_code = 429
    code = "AI_RATE_LIMITED"
    public_message = "Too many AI requests. Please try again later."

    # Gắn Retry-After để client biết thời điểm hợp lý tiếp theo thay vì retry liên tục.
    def __init__(self, retry_after_seconds: int) -> None:
        self.headers = {"Retry-After": str(retry_after_seconds)}
        super().__init__(self.public_message)
