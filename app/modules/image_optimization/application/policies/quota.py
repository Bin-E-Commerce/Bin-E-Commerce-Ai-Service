"""Helpers for the image optimization quota contract."""

from uuid import UUID


def image_optimization_rate_limit_key(seller_owner_id: UUID) -> str:
    """Build the stable, non-sensitive limiter key shared by mutations and overview reads."""

    return f"ai:image-optimization:{seller_owner_id}"
