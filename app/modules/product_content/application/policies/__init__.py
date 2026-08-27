"""Công khai application policies của product content."""

from app.modules.product_content.application.policies.cdn_images import validate_cdn_images

__all__ = ["validate_cdn_images"]
