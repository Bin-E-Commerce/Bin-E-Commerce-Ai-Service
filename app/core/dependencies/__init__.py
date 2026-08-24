"""Public boundary cho dependency injection của FastAPI."""

from app.core.dependencies.providers import get_current_user, get_product_name_service

__all__ = ["get_current_user", "get_product_name_service"]
