"""FastAPI application factory và cross-cutting HTTP handlers.

File chỉ compose router, middleware và stable error envelope; không chứa business
rule, provider call hoặc database query.
"""

import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.bootstrap.http_metrics import record_http_request_metrics
from app.bootstrap.lifespan import application_lifespan
from app.core.errors import AppError
from app.core.metrics import MetricsRegistry
from app.modules.image_optimization.presentation.api.router import router as image_optimization_router
from app.modules.product_content.presentation.api.router import router as product_content_router
from app.modules.ranking.presentation.api.router import router as ranking_router
from app.modules.search_ranking.image_embedding.presentation.router import router as search_image_embedding_router
from app.modules.search_ranking.presentation.router import router as search_ranking_router

logger = logging.getLogger("app.http")


# Tạo correlation ID an toàn từ header Gateway hoặc UUID mới.
def _request_id(request: Request) -> str:
    """Giới hạn 128 ký tự để header không bị lợi dụng làm phình log/response."""

    candidate = request.headers.get("x-request-id", "").strip()
    return candidate[:128] if candidate else str(uuid4())


# Tạo FastAPI app dùng cùng wiring path cho production và test.
def create_application() -> FastAPI:
    """Đăng ký error handler trước router để mọi AppError có cùng envelope."""

    application = FastAPI(title="Bin AI Service", version="0.2.0", lifespan=application_lifespan)
    application.state.metrics = MetricsRegistry()

    # Map AppError thành payload ổn định và không trả exception detail nội bộ.
    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        """Giữ status/code/message/requestId nhất quán trên mọi module."""

        correlation_id = getattr(request.state, "request_id", _request_id(request))
        content = {"code": error.code, "message": error.public_message, "requestId": correlation_id}
        if error.details:
            content["details"] = error.details
        return JSONResponse(
            status_code=error.status_code,
            content=content,
            headers={**error.headers, "x-request-id": correlation_id},
        )

    # Gắn request ID trước khi route/use case chạy và phản hồi lại cho Gateway.
    @application.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Không log payload hoặc user identity; middleware chỉ quản lý correlation ID."""

        request.state.request_id = _request_id(request)
        started_at = monotonic()
        metrics = application.state.metrics
        metrics.add_gauge("http_requests_in_flight", 1, {"service": "ai-service"})
        try:
            response = await call_next(request)
        except Exception as error:
            # Exception chưa thành Response vẫn được tính là 5xx; không ghi exception message có thể chứa input.
            duration_seconds = monotonic() - started_at
            route = request.scope.get("route")
            labels = record_http_request_metrics(
                metrics,
                request.method,
                getattr(route, "path", None),
                500,
                duration_seconds,
            )
            logger.warning(
                "http.request.failed",
                extra={
                    "request_id": request.state.request_id,
                    "method": labels["method"],
                    "route": labels["route"],
                    "status_code": 500,
                    "exception_type": error.__class__.__name__,
                    "duration_ms": round(duration_seconds * 1000),
                },
            )
            raise
        finally:
            # Kể cả request bị exception hoặc cancellation, gauge phải trở về đúng số request còn chạy.
            metrics.add_gauge("http_requests_in_flight", -1, {"service": "ai-service"})

        duration_seconds = monotonic() - started_at
        route = request.scope.get("route")
        labels = record_http_request_metrics(
            metrics,
            request.method,
            getattr(route, "path", None),
            response.status_code,
            duration_seconds,
        )
        if response.status_code >= 400 or duration_seconds >= 0.5:
            # Chỉ ghi request lỗi/chậm để Loki không bị ngập bởi health probe vài giây một lần.
            logger.warning(
                "http.request.completed",
                extra={
                    "request_id": request.state.request_id,
                    "method": labels["method"],
                    "route": labels["route"],
                    "status_code": response.status_code,
                    "duration_ms": round(duration_seconds * 1000),
                },
            )
        response.headers["x-request-id"] = request.state.request_id
        return response

    # Liveness endpoint không gọi provider trả phí hoặc dependency bên ngoài.
    @application.get("/api/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """Chỉ xác nhận API process đang nhận request."""

        return {"status": "ok", "service": "ai-service"}

    # Endpoint chỉ trả metric vận hành có cardinality thấp, không trả raw prompt hoặc provider data.
    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Xuất metrics cho Prometheus mà không gọi dependency bên ngoài."""

        return Response(
            content=application.state.metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    application.include_router(product_content_router)
    application.include_router(image_optimization_router)
    application.include_router(ranking_router)
    application.include_router(search_ranking_router)
    application.include_router(search_image_embedding_router)
    return application
