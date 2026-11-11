"""Kiểm thử split thời gian và query group của pipeline training."""

from datetime import UTC, datetime

import pytest

from app.modules.ranking.infrastructure.training.lightgbm_trainer import (
    _group_for_ranking,
    _monotone_constraints,
    _validate_feature_schema_version,
    _validate_temporal_order,
)
from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RankingEvaluationRow,
)


# Tạo cặp candidate có cùng thời gian request để test split không cần model hoặc dataset thật.
def _request_rows(
    request_id: str,
    request_time: str,
) -> list[RankingEvaluationRow]:
    """Trả hai candidate có nhãn khác nhau trong một ranking query."""

    timestamp = datetime.fromisoformat(request_time).astimezone(UTC)
    return [
        RankingEvaluationRow([0.8] + [0.0] * 8, 1.0, request_id, timestamp),
        RankingEvaluationRow([0.2] + [0.0] * 8, 0.0, request_id, timestamp),
    ]


# profileAffinity chỉ tăng/giữ score và negativePenalty chỉ giảm/giữ score trong model LambdaRank.
def test_monotone_constraints_follow_feature_names_after_ablation() -> None:
    """Constraint dương/âm gắn đúng feature và biến mất cùng feature bị loại."""

    assert _monotone_constraints() == [1, 0, 0, 0, 0, 0, 0, 0, -1]
    assert _monotone_constraints(tuple(range(1, 9))) == [0] * 7 + [-1]
    assert _monotone_constraints(tuple(range(8))) == [1] + [0] * 7


# Manifest có cùng chín cột nhưng semantics phiên bản cũ không được dùng để train artifact mới.
def test_feature_schema_version_rejects_old_or_missing_manifest_value() -> None:
    """Trainer chỉ nhận chính xác feature schema version đang triển khai."""

    assert (
        _validate_feature_schema_version(
            "ranking-features-v2",
            "ranking-features-v2",
        )
        == "ranking-features-v2"
    )
    with pytest.raises(ValueError, match="FEATURE_SCHEMA_VERSION_MISMATCH"):
        _validate_feature_schema_version("ranking-features-v1", "ranking-features-v2")


# LambdaRank cần query candidates liên tiếp và group size tổng khớp số row.
def test_group_for_ranking_keeps_candidates_together() -> None:
    """Group array mô tả kích thước request theo thứ tự xuất hiện."""

    rows = _request_rows("request-a", "2026-01-01T00:00:00+00:00")
    rows += _request_rows("request-b", "2026-01-02T00:00:00+00:00")

    ordered, groups = _group_for_ranking(rows)

    assert [row.request_id for row in ordered] == [
        "request-a",
        "request-a",
        "request-b",
        "request-b",
    ]
    assert groups == [2, 2]
    assert sum(groups) == len(ordered)


# Split temporal hợp lệ phải có cutoff tăng nghiêm ngặt và không lặp request.
def test_temporal_validation_accepts_strictly_ordered_splits() -> None:
    """Train xảy ra trước validation, validation trước test."""

    _validate_temporal_order(
        _request_rows("train-1", "2026-01-01T00:00:00+00:00"),
        _request_rows("validation-1", "2026-01-02T00:00:00+00:00"),
        _request_rows("test-1", "2026-01-03T00:00:00+00:00"),
    )


# Thời gian chồng lấn giữa train và validation làm mất ý nghĩa đánh giá tương lai.
def test_temporal_validation_rejects_non_chronological_splits() -> None:
    """Validation phải bắt đầu sau thời điểm request train cuối cùng."""

    with pytest.raises(ValueError, match="SPLITS_MUST_BE_STRICTLY_CHRONOLOGICAL"):
        _validate_temporal_order(
            _request_rows("train-1", "2026-01-02T00:00:00+00:00"),
            _request_rows("validation-1", "2026-01-01T00:00:00+00:00"),
            _request_rows("test-1", "2026-01-03T00:00:00+00:00"),
        )


# Cùng request xuất hiện ở nhiều split phải bị từ chối trước fitting.
def test_temporal_validation_rejects_query_leakage() -> None:
    """Không chia candidate của cùng recommendation request sang hai tập."""

    with pytest.raises(ValueError, match="REQUEST_GROUP_LEAKAGE_ACROSS_SPLITS"):
        _validate_temporal_order(
            _request_rows("shared-request", "2026-01-01T00:00:00+00:00"),
            _request_rows("shared-request", "2026-01-02T00:00:00+00:00"),
            _request_rows("test-1", "2026-01-03T00:00:00+00:00"),
        )
