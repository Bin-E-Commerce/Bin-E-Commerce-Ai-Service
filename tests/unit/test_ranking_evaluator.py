"""Kiểm thử metric offline cho dataset ranking synthetic."""

import pytest

from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RankingEvaluationRow,
    binary_auc,
    blended_ranking_predictions,
    evaluate_predictions,
    standard_ranking_predictions,
)


# Tạo group nhỏ có positive đứng đầu khi prediction tốt để kiểm tra metric ranking.
def _rows() -> list[RankingEvaluationRow]:
    """Trả về hai request group, mỗi group có một candidate dương và âm."""

    return [
        RankingEvaluationRow([0.9, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, "request-a"),
        RankingEvaluationRow([0.1, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, "request-a"),
        RankingEvaluationRow([0.8, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, "request-b"),
        RankingEvaluationRow([0.2, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, "request-b"),
    ]


# Prediction xếp đúng candidate positive phải đạt các metric ranking tối đa.
def test_evaluate_predictions_returns_perfect_ranking_metrics() -> None:
    """AUC, NDCG và MRR bằng một khi thứ tự từng request là hoàn hảo."""

    metrics = evaluate_predictions(_rows(), [0.95, 0.05, 0.9, 0.1])

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["ndcgAt5"] == pytest.approx(1.0)
    assert metrics["ndcgAt10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["positiveRate"] == pytest.approx(0.5)


# Metric global phải trả None cho AUC khi dữ liệu chỉ có một class.
def test_binary_auc_returns_none_for_single_class() -> None:
    """Không giả tạo AUC khi không có cả positive và negative để so sánh."""

    assert binary_auc([1.0, 1.0], [0.2, 0.8]) is None


# NDCG/MRR cần requestId để nhóm candidate; dataset legacy vẫn được đánh giá binary.
def test_evaluate_predictions_handles_missing_request_groups() -> None:
    """Legacy rows không có requestId vẫn có log loss và AUC hợp lệ."""

    rows = [
        RankingEvaluationRow([0.9, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, None),
        RankingEvaluationRow([0.1, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, None),
    ]

    metrics = evaluate_predictions(rows, [0.9, 0.1])

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["ndcgAt5"] is None
    assert metrics["ndcgAt10"] is None
    assert metrics["mrr"] is None


# Mọi metric phải từ chối prediction lệch số dòng để tránh báo cáo sai artifact.
def test_evaluate_predictions_rejects_length_mismatch() -> None:
    """Prediction batch không cùng kích thước với rows là lỗi contract."""

    with pytest.raises(ValueError, match="METRIC_INPUT_LENGTH_MISMATCH"):
        evaluate_predictions(_rows(), [0.5])


# Baseline và blend phải giữ score bounded để đồng nhất với response serving.
def test_standard_and_blended_predictions_are_bounded() -> None:
    """Comparator không tạo score ngoài [0, 1] dù input model có giá trị cực biên."""

    standard = standard_ranking_predictions(_rows())
    blended = blended_ranking_predictions(standard, [1.0, 0.0, 1.0, 0.0])

    assert all(0.0 <= score <= 1.0 for score in standard)
    assert all(0.0 <= score <= 1.0 for score in blended)
