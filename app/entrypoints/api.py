"""Entrypoint FastAPI; chỉ tạo application từ composition root."""

from app.bootstrap.api import create_application

app = create_application()

__all__ = ["app"]
