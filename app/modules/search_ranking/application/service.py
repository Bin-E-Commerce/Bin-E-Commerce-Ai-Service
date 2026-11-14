"""Use case batch reranking, độc lập với HTTP, ONNX Runtime và persistence."""

import math
from collections.abc import Sequence
from typing import Protocol

from app.core.errors import InvalidInputError, InvalidProviderResponseError
from app.modules.search_ranking.domain.models import (
    SearchRerankBatch,
    SearchRerankPrediction,
)


class SearchRerankerModel(Protocol):
    """Contract tối thiểu của artifact Cross-Encoder đã nạp."""

    model_version: str

    def predict(self, query: str, documents: Sequence[str]) -> list[float]:
        """Chấm điểm relevance theo từng cặp query-document cùng thứ tự đầu vào."""


class SearchRerankingService:
    """Chuẩn hóa input giới hạn và map điểm model về candidate ID."""

    def __init__(self, model: SearchRerankerModel, max_items: int, max_product_chars: int) -> None:
        self._model = model
        self._max_items = max_items
        self._max_product_chars = max_product_chars

    # Chặn dữ liệu quá lớn hoặc ID trùng trước khi tokenize để giới hạn CPU và tránh map nhầm candidate.
    def rerank(
        self,
        request_id: str,
        query: str,
        items: Sequence[tuple[str, str]],
    ) -> SearchRerankBatch:
        """Trả raw relevance score; caller quyết định tie-break và áp dụng hard business rules."""

        normalized_query = query.strip()
        normalized_request_id = request_id.strip()[:128]
        if (
            not normalized_query
            or len(normalized_query) > 120
            or not normalized_request_id
            or not items
            or len(items) > self._max_items
        ):
            raise InvalidInputError()

        seen_ids: set[str] = set()
        item_ids: list[str] = []
        documents: list[str] = []
        for item_id, raw_document in items:
            normalized_id = item_id.strip()
            document = raw_document.strip()
            if (
                not normalized_id
                or len(normalized_id) > 128
                or normalized_id in seen_ids
                or not document
                or len(document) > self._max_product_chars
            ):
                raise InvalidInputError()
            seen_ids.add(normalized_id)
            item_ids.append(normalized_id)
            documents.append(document)

        scores = self._model.predict(normalized_query, documents)
        if len(scores) != len(item_ids) or not all(math.isfinite(float(score)) for score in scores):
            raise InvalidProviderResponseError()
        predictions = tuple(
            SearchRerankPrediction(item_id=item_id, score=float(score)) for item_id, score in zip(item_ids, scores, strict=True)
        )
        return SearchRerankBatch(
            request_id=normalized_request_id,
            model_version=self._model.model_version,
            predictions=predictions,
        )
