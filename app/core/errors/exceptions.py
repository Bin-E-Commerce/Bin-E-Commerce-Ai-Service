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


# Phân biệt thiếu authentication với thiếu permission để Gateway/client không xử lý nhầm 401 và 403.
class AuthenticationError(AppError):
    """Identity nội bộ hoặc context xác thực không được upstream chấp nhận."""

    status_code = 401
    code = "UNAUTHORIZED"
    public_message = "Authentication is required."


# Trả 404 ổn định khi Product/Media Service xác nhận resource không tồn tại hoặc không còn khả dụng.
class ResourceNotFoundError(AppError):
    """Không làm lộ resource của seller khác; chỉ phản ánh resource hiện tại không dùng được."""

    status_code = 404
    code = "AI_RESOURCE_NOT_FOUND"
    public_message = "The requested AI resource was not found."


# Giữ conflict downstream, đặc biệt optimistic product version, thay vì biến mọi lỗi thành 503.
class UpstreamConflictError(AppError):
    """Downstream từ chối mutation do version hoặc idempotency conflict."""

    status_code = 409
    code = "AI_UPSTREAM_CONFLICT"
    public_message = "The resource changed while the AI request was being processed."


# Map request 4xx không thuộc auth/not-found/conflict sang lỗi client ổn định.
class UpstreamRequestError(AppError):
    """Payload đã qua AI Service nhưng downstream không thể chấp nhận theo contract hiện tại."""

    status_code = 400
    code = "AI_UPSTREAM_REQUEST_REJECTED"
    public_message = "The downstream service rejected the AI request."


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


# Phân biệt lỗi quyền/cấu hình provider với lỗi mạng để vận hành không phải đoán nguyên nhân từ mã chung.
class ProviderConfigurationError(AppError):
    """Provider từ chối vì API key, project hoặc quyền dùng model chưa hợp lệ."""

    status_code = 503
    code = "AI_PROVIDER_CONFIGURATION_ERROR"
    public_message = "The AI provider is not enabled for this project."


# Báo riêng quota/rate limit của provider; không nhầm với rate limit do seller đặt ở AI Service.
class ProviderRateLimitedError(AppError):
    """Provider tạm thời giới hạn số lượt tạo ảnh."""

    status_code = 503
    code = "AI_PROVIDER_RATE_LIMITED"
    public_message = "The AI image provider is temporarily rate limited."


# Báo timeout riêng để UI phân biệt provider chậm với lỗi cấu hình hoặc request bị từ chối.
class ProviderTimeoutError(AppError):
    """Provider không trả kết quả trong thời gian profile cho phép."""

    status_code = 503
    code = "AI_PROVIDER_TIMEOUT"
    public_message = "The AI image provider took too long to respond."


# Báo request không hợp lệ ở provider để tránh retry một lời gọi trả phí chắc chắn thất bại.
class ProviderRequestRejectedError(AppError):
    """Provider không chấp nhận tham số hoặc dữ liệu ảnh của request."""

    status_code = 502
    code = "AI_PROVIDER_REQUEST_REJECTED"
    public_message = "The AI image provider rejected the request."


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


# Báo conflict khi client dùng lại một idempotency key cho payload khác.
class IdempotencyKeyReusedError(AppError):
    """Ngăn trả nhầm batch cũ và ngăn tạo thêm lời gọi provider ngoài ý muốn."""

    status_code = 409
    code = "IDEMPOTENCY_KEY_REUSED"
    public_message = "The idempotency key was already used for a different request."


# Báo conflict khi output seller chọn không thuộc job hoặc job chưa thể apply.
class OptimizationJobNotReadyError(AppError):
    """Giữ error contract ổn định cho các lỗi lifecycle và output selection."""

    status_code = 409
    code = "AI_IMAGE_OPTIMIZATION_JOB_NOT_READY"
    public_message = "The image optimization job is not ready to apply."
