"""ASGI entrypoint cho AI Service API.

Mọi wiring nằm trong `app.bootstrap`; file này chỉ export `app` và factory tương
thích với test/uvicorn hiện tại.
"""

from fastapi import FastAPI

from app.bootstrap.api import create_application
from app.entrypoints.api import app


# Giữ factory công khai hiện tại để test và tooling không phải đổi import.
def create_app() -> FastAPI:
    """Tạo một FastAPI application độc lập cho mỗi test/process."""

    return create_application()


__all__ = ["app", "create_app"]
