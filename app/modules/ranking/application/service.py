"""Use case batch ranking prediction với giới hạn kích thước và output bounded."""

import math
from collections.abc import Sequence

from app.core.errors import InvalidInputError, InvalidProviderResponseError
from app.modules.ranking.application.ports import RankingModel
from app.modules.ranking.domain.models import RankingPrediction, RankingPredictionBatch


class RankingPredictionService:
    """Validate feature batch, gọi model và map score về contract ổn định."""

    def __init__(
        self,
        model: RankingModel,
        max_items: int = 300,
        max_features: int = 64,
        expected_features: int | None = None,
    ) -> None:
        """Nhận model qua port để endpoint test được mà không load artifact thật."""

        self._model = model
        self._max_items = max(1, max_items)
        self._max_features = max(1, max_features)
        self._expected_features = expected_features if expected_features and expected_features > 0 else None

    # Chặn payload quá lớn và giá trị NaN/Infinity trước khi chạy model để bảo vệ CPU/response contract.
    def predict(self, request_id: str, items: Sequence[tuple[str, Sequence[float]]]) -> RankingPredictionBatch:
        """Dự đoán một batch bounded và giữ nguyên thứ tự item input."""

        if not request_id.strip() or not items or len(items) > self._max_items:
            raise InvalidInputError()
        rows: list[list[float]] = []
        item_ids: list[str] = []
        feature_count: int | None = None
        seen_item_ids: set[str] = set()
        for item_id, features in items:
            normalized_item_id = item_id.strip()
            if (
                not normalized_item_id
                or normalized_item_id in seen_item_ids
                or not features
                or len(features) > self._max_features
                or (self._expected_features is not None and len(features) != self._expected_features)
            ):
                raise InvalidInputError()
            row = [float(value) for value in features]
            if not all(math.isfinite(value) for value in row):
                raise InvalidInputError()
            if feature_count is None:
                feature_count = len(row)
            elif feature_count != len(row):
                raise InvalidInputError()
            seen_item_ids.add(normalized_item_id)
            item_ids.append(normalized_item_id)
            rows.append(row)
        scores = self._model.predict(rows)
        if len(scores) != len(item_ids) or not all(math.isfinite(float(score)) for score in scores):
            raise InvalidProviderResponseError()
        predictions = tuple(
            RankingPrediction(
                item_id=item_id,
                # Giới hạn score ở application boundary để model artifact không thể làm hỏng contract ranking.
                score=max(0.0, min(1.0, float(score))),
            )
            for item_id, score in zip(item_ids, scores, strict=True)
        )
        return RankingPredictionBatch(
            request_id=request_id.strip()[:128],
            model_version=self._model.model_version,
            predictions=predictions,
        )
