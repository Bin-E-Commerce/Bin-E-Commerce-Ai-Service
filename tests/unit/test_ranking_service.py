"""Unit test cho batch ranking: validate bounded input va giu mapping item score."""

import pytest

from app.core.errors.exceptions import InvalidInputError
from app.modules.ranking.application.service import RankingPredictionService
from app.modules.ranking.infrastructure.model_registry import FallbackRankingModel


class FakeRankingModel:
    """Fake model de test application service ma khong can LightGBM artifact."""

    model_version = "fake-ranking-v1"

    def predict(self, rows: list[list[float]]) -> list[float]:
        """Tra score co dinh theo thu tu input de kiem tra mapping."""

        return [float(index) / 10 for index, _ in enumerate(rows, start=1)]


def test_prediction_keeps_item_order_and_model_version() -> None:
    """Batch response phai map dung item ID va version model."""

    service = RankingPredictionService(FakeRankingModel(), expected_features=2)

    result = service.predict("request-1", [("product-a", [0.1, 0.2]), ("product-b", [0.3, 0.4])])

    assert result.request_id == "request-1"
    assert result.model_version == "fake-ranking-v1"
    assert [item.item_id for item in result.predictions] == ["product-a", "product-b"]
    assert [item.score for item in result.predictions] == [0.1, 0.2]


def test_prediction_rejects_mixed_feature_dimensions() -> None:
    """Khong cho model nhan batch lech so cot vi se lam artifact predict sai."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict("request-1", [("product-a", [0.1, 0.2]), ("product-b", [0.3])])


def test_prediction_rejects_duplicate_item_ids() -> None:
    """Khong cho phep map hai score vao cung mot candidate ID."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict("request-1", [("product-a", [0.1]), (" product-a ", [0.2])])


def test_prediction_normalizes_ids_and_clamps_model_score() -> None:
    """ID duoc trim va score ngoai [0, 1] khong duoc lan vao response."""

    class OutOfRangeModel:
        model_version = "out-of-range-v1"

        def predict(self, rows: list[list[float]]) -> list[float]:
            """Tra score ngoai range de kiem tra application boundary."""

            return [1.5 for _ in rows]

    service = RankingPredictionService(OutOfRangeModel())

    result = service.predict("request-1", [(" product-a ", [0.1])])

    assert result.predictions[0].item_id == "product-a"
    assert result.predictions[0].score == 1.0


def test_fallback_ranking_subtracts_negative_penalty() -> None:
    """Fallback giu y nghia penalty thay vi cong penalty vao diem duong."""

    scores = FallbackRankingModel().predict([[1.0, 1.0, 0.2]])

    assert scores == [0.8]
