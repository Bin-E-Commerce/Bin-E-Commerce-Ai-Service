"""Public exports cho các HTTP client nội bộ của image optimization.

Mỗi adapter phụ trách đúng một upstream capability và dùng chung connection pool.
"""

from .media_assets import HttpMediaAssetClient
from .product_media import HttpProductMediaClient
from .product_owner import HttpProductOwnerClient

__all__ = ["HttpMediaAssetClient", "HttpProductMediaClient", "HttpProductOwnerClient"]
