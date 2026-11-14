"""Kiểu dữ liệu nội bộ và contract phiên bản cho search reranker."""

from dataclasses import dataclass

SEARCH_RERANKER_SCHEMA_VERSION = "search-reranker-v1"


@dataclass(frozen=True, slots=True)
class SearchRerankPrediction:
    """Điểm relevance thô cho một sản phẩm trong cùng truy vấn."""

    item_id: str
    score: float


@dataclass(frozen=True, slots=True)
class SearchRerankBatch:
    """Kết quả một request giữ nguyên mã phiên bản artifact đang phục vụ."""

    request_id: str
    model_version: str
    predictions: tuple[SearchRerankPrediction, ...]
