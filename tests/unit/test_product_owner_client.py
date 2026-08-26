"""Kiểm tra adapter Product Service xử lý product cũ chưa có externalImageId."""

from uuid import UUID

from app.modules.image_optimization.infrastructure.clients import HttpProductOwnerClient


# Kiểm tra fallback chỉ chấp nhận URL HTTPS có owner và asset UUID khớp dữ liệu media chuẩn.
def test_extracts_asset_id_from_owned_processed_product_url() -> None:
    """Lấy đúng asset ID từ URL CDN của seller hiện tại."""

    owner_id = UUID("226123a7-91de-4738-b7b1-c39f1caab780")
    media_url = (
        "https://cdn.example.com/media/processed/product_image/"
        "226123a7-91de-4738-b7b1-c39f1caab780/7ac6fabe-44db-456b-baed-3733f7e898f0/large.webp"
    )

    assert HttpProductOwnerClient._asset_id_from_product_url(media_url, owner_id) == ("7ac6fabe-44db-456b-baed-3733f7e898f0")


# URL khác owner phải bị loại bỏ để tránh worker đọc nhầm media của shop khác.
def test_rejects_processed_product_url_from_another_owner() -> None:
    """Không suy ra asset ID khi owner trong URL không trùng seller hiện tại."""

    owner_id = UUID("226123a7-91de-4738-b7b1-c39f1caab780")
    media_url = (
        "https://cdn.example.com/media/processed/product_image/different-owner/7ac6fabe-44db-456b-baed-3733f7e898f0/large.webp"
    )

    assert HttpProductOwnerClient._asset_id_from_product_url(media_url, owner_id) is None
