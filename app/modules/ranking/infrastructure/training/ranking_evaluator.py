"""Đánh giá ranking offline theo từng request mà không gọi dịch vụ bên ngoài.

Module chỉ nhận label, raw score và request group. Các phép tính này phục vụ
so sánh LambdaRank với baseline kiểm chứng; không được gọi trong request path.
"""

import math
from dataclasses import dataclass
from datetime import datetime

# Thứ tự phải khớp FEATURE_ORDER trong RecommendationMlRankingService.
RANKING_FEATURE_NAMES = (
    "profileAffinity",
    "sessionContext",
    "semanticSimilarity",
    "coBehavior",
    "popularity",
    "freshness",
    "quality",
    "exploration",
    "negativePenalty",
)
PROFILE_AFFINITY_FEATURE_INDEX = RANKING_FEATURE_NAMES.index("profileAffinity")
POPULARITY_FEATURE_INDEX = RANKING_FEATURE_NAMES.index("popularity")


# Dữ liệu bất biến đại diện một candidate và outcome trong một request ranking.
@dataclass(frozen=True)
class RankingEvaluationRow:
    """Một row đã chuẩn hóa, có feature, relevance label và request group."""

    features: list[float]
    label: float
    request_id: str | None
    request_time: datetime | None = None


# Metric có thể không xác định khi thiếu request group hoặc không có positive.
MetricValue = float | None


# Tính AUC theo rank của score raw; metric này đo khả năng phân biệt cặp positive/negative toàn cục.
def binary_auc(labels: list[float], predictions: list[float]) -> MetricValue:
    """Trả AUC hoặc None khi tập dữ liệu không có cả hai lớp."""

    # Kích thước sai sẽ ghép nhầm nhãn với score; tập rỗng không có metric.
    if len(labels) != len(predictions) or not labels:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError("METRIC_PREDICTIONS_MUST_BE_FINITE")
    positives = sum(1 for label in labels if label > 0)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    # Sort tăng dần và gán rank trung bình cho tie để AUC không phụ thuộc thứ tự đầu vào.
    ordered = sorted(enumerate(predictions), key=lambda item: item[1])
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        positive_rank_sum += sum(average_rank for index, _ in ordered[cursor:end] if labels[index] > 0)
        cursor = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


# Gom candidate theo request vì các metric top-K chỉ có nghĩa trong cùng một danh sách được phục vụ.
def _group_rows(rows: list[RankingEvaluationRow], predictions: list[float]) -> dict[str, list[tuple[float, float]]]:
    """Ghép raw score/label vào request group, bỏ row thiếu request ID."""

    # Giữ cặp row-score theo vị trí gốc; thiếu request ID không thể được xem như ranking query.
    if len(rows) != len(predictions) or not rows:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError("METRIC_PREDICTIONS_MUST_BE_FINITE")
    groups: dict[str, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row.request_id:
            groups.setdefault(row.request_id, []).append((prediction, row.label))
    return groups


# Tính DCG/IDCG theo binary relevance; item positive ở rank cao đóng góp nhiều hơn.
def ndcg_at_k(
    rows: list[RankingEvaluationRow],
    predictions: list[float],
    k: int,
) -> MetricValue:
    """Trả NDCG@k macro-average trên các request có ID."""

    # k không dương không phải một ranking cutoff; không tạo metric giả từ slice rỗng.
    if k < 1:
        raise ValueError("METRIC_CUTOFF_MUST_BE_POSITIVE")
    groups = _group_rows(rows, predictions)
    if not groups:
        return None
    scores: list[float] = []
    for group in groups.values():
        # Dùng đúng top-k model score và so với thứ tự lý tưởng của cùng request.
        ranked = sorted(group, key=lambda item: item[0], reverse=True)[:k]
        ideal = sorted((label for _, label in group), reverse=True)[:k]
        dcg = sum(label / math.log2(index + 2) for index, (_, label) in enumerate(ranked))
        idcg = sum(label / math.log2(index + 2) for index, label in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)


# Tính Recall@k theo mỗi request; request không có positive không vào mẫu số macro-average.
def recall_at_k(
    rows: list[RankingEvaluationRow],
    predictions: list[float],
    k: int,
) -> MetricValue:
    """Trả tỷ lệ positive được tìm thấy trong top-k trung bình theo request."""

    # Kiểm tra cutoff trước khi nhóm để cùng contract với NDCG.
    if k < 1:
        raise ValueError("METRIC_CUTOFF_MUST_BE_POSITIVE")
    groups = _group_rows(rows, predictions)
    if not groups:
        return None
    recalls: list[float] = []
    for group in groups.values():
        # Query không có positive không thể đánh giá recall nên được loại khỏi macro-average.
        relevant_count = sum(1 for _, label in group if label > 0)
        if relevant_count == 0:
            continue
        ranked = sorted(group, key=lambda item: item[0], reverse=True)[:k]
        retrieved_count = sum(1 for _, label in ranked if label > 0)
        recalls.append(retrieved_count / relevant_count)
    return sum(recalls) / len(recalls) if recalls else 0.0


# Tính MRR theo rank positive đầu tiên trong mỗi request có positive.
def mean_reciprocal_rank(rows: list[RankingEvaluationRow], predictions: list[float]) -> MetricValue:
    """Trả MRR macro-average hoặc None nếu thiếu request ID."""

    groups = _group_rows(rows, predictions)
    if not groups:
        return None
    reciprocal_ranks: list[float] = []
    for group in groups.values():
        # Chỉ request có positive mới xác định được vị trí kết quả liên quan đầu tiên.
        ranked = sorted(group, key=lambda item: item[0], reverse=True)
        first_positive = next(
            (index + 1 for index, (_, label) in enumerate(ranked) if label > 0),
            None,
        )
        if first_positive is not None:
            reciprocal_ranks.append(1.0 / first_positive)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# Gom metric ranking trên đúng một batch score để tránh report các chỉ số từ các lần predict khác nhau.
def evaluate_predictions(rows: list[RankingEvaluationRow], predictions: list[float]) -> dict[str, MetricValue]:
    """Đánh giá raw LambdaRank score bằng metric phân loại-thứ tự và top-K."""

    # Chặn độ dài lệch trước khi tính positiveRate độc lập với helper ranking.
    if len(rows) != len(predictions) or not rows:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    labels = [row.label for row in rows]
    return {
        "auc": binary_auc(labels, predictions),
        "ndcgAt5": ndcg_at_k(rows, predictions, 5),
        "ndcgAt10": ndcg_at_k(rows, predictions, 10),
        "recallAt5": recall_at_k(rows, predictions, 5),
        "recallAt10": recall_at_k(rows, predictions, 10),
        "mrr": mean_reciprocal_rank(rows, predictions),
        "positiveRate": sum(1 for label in labels if label > 0) / len(labels),
    }


# Dùng riêng profileAffinity để định lượng khả năng của tiêu chí đang nghiên cứu, không dùng ở serving.
def profile_affinity_baseline_predictions(
    rows: list[RankingEvaluationRow],
) -> list[float]:
    """Xếp candidate theo feature profileAffinity đã tính từ hồ sơ."""

    # Input feature đã bounded bởi Recommendation Service; lỗi thiếu cột không được biến thành 0 âm thầm.
    if any(len(row.features) <= PROFILE_AFFINITY_FEATURE_INDEX for row in rows):
        raise ValueError("PROFILE_AFFINITY_FEATURE_MISSING")
    return [row.features[PROFILE_AFFINITY_FEATURE_INDEX] for row in rows]


# Dùng popularity làm mốc đơn giản để biết model có học vượt tín hiệu phổ biến không.
def popularity_baseline_predictions(
    rows: list[RankingEvaluationRow],
) -> list[float]:
    """Xếp candidate theo feature popularity, chỉ phục vụ báo cáo offline."""

    # Dataset sai schema phải fail sớm để không báo cáo nhầm một cột feature khác thành popularity.
    if any(len(row.features) <= POPULARITY_FEATURE_INDEX for row in rows):
        raise ValueError("POPULARITY_FEATURE_MISSING")
    return [row.features[POPULARITY_FEATURE_INDEX] for row in rows]


# Xếp từng feature đơn lẻ làm baseline để kiểm tra tiêu chí nào có liên hệ với label.
def feature_only_baseline_predictions(rows: list[RankingEvaluationRow], feature_name: str) -> list[float]:
    """Trả score baseline từ đúng một cột, đảo dấu penalty âm."""

    try:
        feature_index = RANKING_FEATURE_NAMES.index(feature_name)
    except ValueError as error:
        raise ValueError("UNKNOWN_RANKING_FEATURE") from error
    if any(len(row.features) <= feature_index for row in rows):
        raise ValueError("RANKING_FEATURE_MISSING")
    direction = -1.0 if feature_name == "negativePenalty" else 1.0
    return [direction * row.features[feature_index] for row in rows]


# Tóm tắt độ rộng từng request để biết NDCG@10 đang đo top-10 hay gần như toàn bộ candidate.
def query_group_diagnostics(
    rows: list[RankingEvaluationRow],
) -> dict[str, int | float | None]:
    """Báo số query và phân phối số candidate trên mỗi query."""

    group_sizes: dict[str, int] = {}
    for row in rows:
        if row.request_id:
            group_sizes[row.request_id] = group_sizes.get(row.request_id, 0) + 1
    sizes = list(group_sizes.values())
    if not sizes:
        return {
            "queryCount": 0,
            "meanCandidatesPerQuery": None,
            "medianCandidatesPerQuery": None,
            "minCandidatesPerQuery": None,
            "maxCandidatesPerQuery": None,
            "queriesWithAtLeast10Candidates": 0,
        }
    ordered_sizes = sorted(sizes)
    midpoint = len(ordered_sizes) // 2
    median_size = (
        float(ordered_sizes[midpoint]) if len(ordered_sizes) % 2 else (ordered_sizes[midpoint - 1] + ordered_sizes[midpoint]) / 2
    )
    return {
        "queryCount": len(sizes),
        "meanCandidatesPerQuery": sum(sizes) / len(sizes),
        "medianCandidatesPerQuery": median_size,
        "minCandidatesPerQuery": min(sizes),
        "maxCandidatesPerQuery": max(sizes),
        "queriesWithAtLeast10Candidates": sum(size >= 10 for size in sizes),
    }
