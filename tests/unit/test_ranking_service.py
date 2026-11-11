"""Kiểm thử contract batch cho LambdaRank application service."""

import pytest

from app.core.errors.exceptions import InvalidInputError
from app.modules.ranking.application.service import RankingPredictionService


# Model giả trả các điểm raw có thể âm hoặc lớn hơn một để kiểm tra mapping theo item.
class FakeRankingModel:
    """Model test xác định, không tải artifact hoặc gọi dịch vụ bên ngoài."""

    model_version = "fake-lambdarank-v1"

    def predict(self, rows: list[list[float]]) -> list[float]:
        """Giữ thứ tự row và trả raw score phục vụ kiểm thử contract."""

        return [-0.4 + index for index, _ in enumerate(rows)]


# Prediction phải bảo toàn score raw và vị trí candidate tương ứng.
def test_prediction_keeps_item_order_raw_score_and_model_version() -> None:
    """LambdaRank score không bị clamp thành xác suất trong [0, 1]."""

    service = RankingPredictionService(FakeRankingModel(), expected_features=2)

    result = service.predict(
        "request-1",
        [("product-a", [0.1, 0.2]), ("product-b", [0.3, 0.4])],
    )

    assert result.request_id == "request-1"
    assert result.model_version == "fake-lambdarank-v1"
    assert [item.item_id for item in result.predictions] == [
        "product-a",
        "product-b",
    ]
    assert [item.score for item in result.predictions] == [-0.4, 0.6]


# Batch có số chiều không đồng nhất không thể dùng một artifact feature schema duy nhất.
def test_prediction_rejects_mixed_feature_dimensions() -> None:
    """Input lệch số feature bị từ chối trước model inference."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict(
            "request-1",
            [("product-a", [0.1, 0.2]), ("product-b", [0.3])],
        )


# Trùng ID làm response không thể ánh xạ duy nhất score về candidate.
def test_prediction_rejects_duplicate_item_ids() -> None:
    """ID sau chuẩn hóa phải duy nhất trong một batch."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict(
            "request-1",
            [("product-a", [0.1]), (" product-a ", [0.2])],
        )


# Model score raw ngoài [0, 1] là hợp lệ; chỉ feature đầu vào mới bị giới hạn.
def test_prediction_preserves_raw_scores_outside_probability_range() -> None:
    """Score 1.5 được giữ nguyên vì LambdaRank chỉ cần thứ tự tương đối."""

    class OutOfRangeModel:
        model_version = "out-of-range-v1"

        def predict(self, rows: list[list[float]]) -> list[float]:
            """Trả score lớn hơn một để bảo vệ contract raw-score."""

            return [1.5 for _ in rows]

    service = RankingPredictionService(OutOfRangeModel())

    result = service.predict("request-1", [(" product-a ", [0.1])])

    assert result.predictions[0].item_id == "product-a"
    assert result.predictions[0].score == 1.5


# Input feature NaN/Infinity bị chặn để model và ranking sort luôn nhận số hữu hạn.
def test_prediction_rejects_non_finite_features() -> None:
    """Giá trị ngoài miền số hữu hạn bị từ chối tại application boundary."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict("request-1", [("product-a", [float("nan")])])


# Model chỉ được nhận feature trong cùng miền đã dùng để huấn luyện và kiểm thử.
def test_prediction_rejects_features_outside_unit_interval() -> None:
    """Feature ngoài [0, 1] không thể đi vào artifact có schema khác."""

    service = RankingPredictionService(FakeRankingModel())

    with pytest.raises(InvalidInputError):
        service.predict("request-1", [("product-a", [1.01])])
