"""Kiểm thử tính đúng đắn của metric và baseline ranking offline."""

import pytest

from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RankingEvaluationRow,
    binary_auc,
    evaluate_predictions,
    feature_only_baseline_predictions,
    ndcg_at_k,
    popularity_baseline_predictions,
    profile_affinity_baseline_predictions,
    query_group_diagnostics,
    recall_at_k,
)


# Tạo hai request có feature profile và popularity khác nhau để kiểm tra đúng cột baseline.
def _rows() -> list[RankingEvaluationRow]:
    """Trả các candidate có nhãn nhị phân và request group ổn định."""

    return [
        RankingEvaluationRow([0.9, 0, 0, 0, 0.2, 0, 0, 0, 0], 1.0, "request-a"),
        RankingEvaluationRow([0.1, 0, 0, 0, 0.8, 0, 0, 0, 0], 0.0, "request-a"),
        RankingEvaluationRow([0.8, 0, 0, 0, 0.3, 0, 0, 0, 0], 1.0, "request-b"),
        RankingEvaluationRow([0.2, 0, 0, 0, 0.7, 0, 0, 0, 0], 0.0, "request-b"),
    ]


# Ranking đúng phải đạt metric hoàn hảo vì mỗi nhóm có một positive đứng trước negative.
def test_evaluate_predictions_returns_perfect_ranking_metrics() -> None:
    """NDCG, Recall, MRR và AUC xác nhận thứ tự đúng theo từng request."""

    metrics = evaluate_predictions(_rows(), [0.95, 0.05, 0.9, 0.1])

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["ndcgAt5"] == pytest.approx(1.0)
    assert metrics["ndcgAt10"] == pytest.approx(1.0)
    assert metrics["recallAt5"] == pytest.approx(1.0)
    assert metrics["recallAt10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["positiveRate"] == pytest.approx(0.5)


# NDCG chiết khấu positive ở rank thấp theo log cơ số 2 đúng định nghĩa DCG.
def test_ndcg_penalizes_a_positive_item_ranked_second() -> None:
    """Một positive đứng thứ hai có NDCG bằng 1/log2(3)."""

    rows = _rows()[:2]

    assert ndcg_at_k(rows, [0.1, 0.9], 2) == pytest.approx(1 / 1.584962500721156)


# Recall@K chỉ đếm positive được lấy trong K vị trí đầu so với tổng positive cùng request.
def test_recall_at_k_uses_each_request_as_the_denominator() -> None:
    """K=1 bỏ lỡ candidate tốt thì recall bằng 0, còn K=2 tìm thấy toàn bộ."""

    rows = _rows()[:2]

    assert recall_at_k(rows, [0.1, 0.9], 1) == pytest.approx(0.0)
    assert recall_at_k(rows, [0.1, 0.9], 2) == pytest.approx(1.0)


# AUC không xác định khi không có cả positive lẫn negative.
def test_binary_auc_returns_none_for_single_class() -> None:
    """Metric không bịa ra kết quả khi dữ liệu chỉ chứa một lớp."""

    assert binary_auc([1.0, 1.0], [0.2, 0.8]) is None


# Các baseline phải đọc đúng feature đã đặt tên và không lẫn chỉ số cột.
def test_baselines_read_their_named_feature_columns() -> None:
    """Profile-only và popularity-only trả đúng cột feature tương ứng."""

    rows = _rows()

    assert profile_affinity_baseline_predictions(rows) == [0.9, 0.1, 0.8, 0.2]
    assert popularity_baseline_predictions(rows) == [0.2, 0.8, 0.3, 0.7]
    assert feature_only_baseline_predictions(rows, "negativePenalty") == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]


# Chẩn đoán group phải cho thấy metric top-10 thực sự có ít nhất mười candidate.
def test_query_group_diagnostics_report_candidate_coverage_at_ten() -> None:
    """Báo độ rộng nhóm để không đọc NDCG@10 sai trên danh sách quá ngắn."""

    rows = _rows() + [RankingEvaluationRow([0.0] * 9, 0.0, "request-c") for _ in range(10)]

    result = query_group_diagnostics(rows)

    assert result["queryCount"] == 3
    assert result["minCandidatesPerQuery"] == 2
    assert result["maxCandidatesPerQuery"] == 10
    assert result["queriesWithAtLeast10Candidates"] == 1


# Không đánh giá một batch có độ dài lệch vì label và prediction sẽ bị ghép sai candidate.
def test_evaluate_predictions_rejects_length_mismatch() -> None:
    """Evaluator fail-fast khi mỗi row không có đúng một score."""

    with pytest.raises(ValueError, match="METRIC_INPUT_LENGTH_MISMATCH"):
        evaluate_predictions(_rows(), [0.5])


# NaN/Infinity không được phép đi vào sort hoặc metric vì thứ tự khi đó không xác định.
def test_evaluate_predictions_rejects_non_finite_scores() -> None:
    """Prediction không hữu hạn bị từ chối trước khi tính metric."""

    with pytest.raises(ValueError, match="METRIC_PREDICTIONS_MUST_BE_FINITE"):
        evaluate_predictions(_rows(), [0.9, float("nan"), 0.8, 0.1])


# Legacy rows không có requestId vẫn có thể dùng AUC, nhưng không tạo metric top-K giả.
def test_evaluate_predictions_handles_missing_request_groups() -> None:
    """Metric theo request là không xác định khi không có requestId."""

    rows = [
        RankingEvaluationRow([0.9, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, None),
        RankingEvaluationRow([0.1, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, None),
    ]

    metrics = evaluate_predictions(rows, [0.9, 0.1])

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["ndcgAt5"] is None
    assert metrics["ndcgAt10"] is None
    assert metrics["recallAt10"] is None
    assert metrics["mrr"] is None
