"""FastAPI application factory và cross-cutting HTTP handlers.

File chỉ compose router, middleware và stable error envelope; không chứa business
rule, provider call hoặc database query.
"""

from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.bootstrap.lifespan import application_lifespan
from app.core.errors import AppError
from app.modules.image_optimization.presentation.api.router import router as image_optimization_router
from app.modules.product_content.presentation.api.router import router as product_content_router


# Tạo correlation ID an toàn từ header Gateway hoặc UUID mới.
def _request_id(request: Request) -> str:
    """Giới hạn 128 ký tự để header không bị lợi dụng làm phình log/response."""

    candidate = request.headers.get("x-request-id", "").strip()
    return candidate[:128] if candidate else str(uuid4())


# Tạo FastAPI app dùng cùng wiring path cho production và test.
def create_application() -> FastAPI:
    """Đăng ký error handler trước router để mọi AppError có cùng envelope."""

    application = FastAPI(title="Bin AI Service", version="0.2.0", lifespan=application_lifespan)

    # Map AppError thành payload ổn định và không trả exception detail nội bộ.
    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        """Giữ status/code/message/requestId nhất quán trên mọi module."""

        correlation_id = getattr(request.state, "request_id", _request_id(request))
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.public_message, "requestId": correlation_id},
            headers={**error.headers, "x-request-id": correlation_id},
        )

    # Gắn request ID trước khi route/use case chạy và phản hồi lại cho Gateway.
    @application.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Không log payload hoặc user identity; middleware chỉ quản lý correlation ID."""

        request.state.request_id = _request_id(request)
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    # Liveness endpoint không gọi provider trả phí hoặc dependency bên ngoài.
    @application.get("/api/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """Chỉ xác nhận API process đang nhận request."""

        return {"status": "ok", "service": "ai-service"}

    application.include_router(product_content_router)
    application.include_router(image_optimization_router)
    return application
