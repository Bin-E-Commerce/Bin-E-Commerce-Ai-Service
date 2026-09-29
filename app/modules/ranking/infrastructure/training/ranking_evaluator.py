"""Đánh giá dataset ranking offline mà không gọi FastAPI, database hay provider bên ngoài.

Module này chỉ nhận label, prediction và request group đã được chuẩn hóa.
Nó không train model, không ghi artifact và không được import vào request path.
"""

import math
from dataclasses import dataclass


# Dòng dữ liệu bất biến đại diện cho một candidate trong một request recommendation.
# `features` là vector 9 số đã được Recommendation Service tạo ra, `label` là kết quả
# hành vi 0/1, còn `request_id` dùng để gom các candidate cùng một lần xếp hạng.
# Evaluator giữ request_id vì AUC nhìn toàn cục, còn NDCG/MRR phải đánh giá theo từng request.
@dataclass(frozen=True)
class RankingEvaluationRow:
    """Một dòng feature/label tối thiểu để tính metric offline theo request."""

    features: list[float]
    label: float
    request_id: str | None


# Metric có thể là số hoặc None khi dữ liệu không đủ điều kiện tính, ví dụ AUC chỉ có một class.
MetricValue = float | None

# Đây là trọng số Standard Ranking đang dùng trong Recommendation Service.
# Thứ tự phải khớp với 8 feature dương đầu tiên; feature thứ 9 là negativePenalty và bị trừ riêng.
STANDARD_RANKING_WEIGHTS = (
    0.25,
    0.18,
    0.15,
    0.10,
    0.12,
    0.08,
    0.08,
    0.04,
)


# Tính LogLoss để đo chất lượng xác suất model dự đoán.
# Prediction càng gần label thật thì LogLoss càng thấp; dự đoán sai nhưng quá tự tin sẽ bị phạt nặng.
# Epsilon chặn log(0), vì xác suất đúng bằng 0 hoặc 1 có thể làm phép tính trở thành vô hạn.
def binary_log_loss(labels: list[float], predictions: list[float]) -> float:
    """Trả log loss trung bình và bảo vệ biên xác suất trước log(0)."""

    # Không được ghép nhầm label của dòng này với prediction của dòng khác.
    if len(labels) != len(predictions) or not labels:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    epsilon = 1e-15
    total = 0.0
    for label, prediction in zip(labels, predictions, strict=True):
        # Ép prediction vào khoảng an toàn trước khi gọi log.
        bounded = min(1.0 - epsilon, max(epsilon, float(prediction)))
        total -= float(label) * math.log(bounded)
        total -= (1.0 - float(label)) * math.log(1.0 - bounded)
    return total / len(labels)


# Tính AUC bằng rank sum thay vì phụ thuộc scikit-learn.
# AUC trả lời câu hỏi: nếu lấy ngẫu nhiên một candidate positive và một candidate negative,
# model có xếp positive cao hơn không? Giá trị 0.5 gần như ngẫu nhiên, càng gần 1 càng tốt.
# Khi tất cả dòng chỉ có một label, không có cặp để so sánh nên trả None.
def binary_auc(labels: list[float], predictions: list[float]) -> MetricValue:
    """Trả AUC hoặc None khi tập dữ liệu chỉ có một class."""

    # Kiểm tra kích thước trước khi tính rank để không tạo metric sai.
    if len(labels) != len(predictions) or not labels:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    positives = sum(1 for label in labels if label > 0)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    # Sắp xếp tăng dần để dùng công thức rank-sum; tie sẽ được gán rank trung bình.
    ordered = sorted(enumerate(predictions), key=lambda item: item[1])
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        positive_rank_sum += sum(
            average_rank for index, _ in ordered[cursor:end] if labels[index] > 0
        )
        cursor = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


# Tính NDCG theo từng request để đo chất lượng thứ tự top-K.
# Một request có nhiều candidate, vì vậy không thể trộn candidate của request này với request khác.
# DCG đo thứ tự model tạo ra; IDCG là thứ tự lý tưởng. NDCG = DCG / IDCG và nằm trong [0, 1].
def ndcg_at_k(
    rows: list[RankingEvaluationRow],
    predictions: list[float],
    k: int,
) -> MetricValue:
    """Trả NDCG@k trung bình trên các request có request_id hợp lệ."""

    # NDCG cần cả row và prediction tương ứng, đồng thời cần ít nhất một row để chia trung bình.
    if len(rows) != len(predictions) or not rows:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    groups: dict[str, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        # Dataset legacy có thể không có request_id; khi đó không đủ thông tin tính ranking metric.
        if row.request_id:
            groups.setdefault(row.request_id, []).append((prediction, row.label))
    if not groups:
        return None
    scores: list[float] = []
    for group in groups.values():
        # Chỉ lấy top-k theo prediction để phản ánh vị trí người dùng thực sự nhìn thấy.
        ranked = sorted(group, key=lambda item: item[0], reverse=True)[:k]
        # Ideal là thứ tự giả định mọi positive đều được đẩy lên đầu.
        ideal = sorted((label for _, label in group), reverse=True)[:k]
        dcg = sum(label / math.log2(index + 2) for index, (_, label) in enumerate(ranked))
        idcg = sum(label / math.log2(index + 2) for index, label in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)


# Tính MRR theo request group.
# MRR chỉ quan tâm vị trí của positive đầu tiên: positive ở vị trí 1 được 1.0,
# ở vị trí 2 được 0.5, ở vị trí 3 được 0.333...; càng cao càng tốt.
def mean_reciprocal_rank(
    rows: list[RankingEvaluationRow],
    predictions: list[float],
) -> MetricValue:
    """Trả MRR trung bình hoặc None khi không có request group."""

    # Nếu prediction lệch row thì thứ hạng được tính cho sai sản phẩm.
    if len(rows) != len(predictions) or not rows:
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    groups: dict[str, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row.request_id:
            groups.setdefault(row.request_id, []).append((prediction, row.label))
    if not groups:
        return None
    reciprocal_ranks: list[float] = []
    for group in groups.values():
        # Xếp candidate giảm dần rồi tìm positive đầu tiên trong nhóm request.
        ranked = sorted(group, key=lambda item: item[0], reverse=True)
        first_positive = next(
            (index + 1 for index, (_, label) in enumerate(ranked) if label > 0),
            None,
        )
        if first_positive is not None:
            reciprocal_ranks.append(1.0 / first_positive)
    return (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )


# Gom tất cả metric thành một contract ổn định cho metadata và báo cáo offline.
# Hàm này được gọi riêng cho validation, test, Standard Ranking, AI Ranking và AI Enhanced Ranking.
def evaluate_predictions(
    rows: list[RankingEvaluationRow],
    predictions: list[float],
) -> dict[str, MetricValue]:
    """Đánh giá binary và ranking metrics trên cùng một prediction batch."""

    # Tách label từ row nhưng vẫn truyền nguyên row vào NDCG/MRR để giữ request group.
    labels = [row.label for row in rows]
    return {
        "logLoss": binary_log_loss(labels, predictions),
        "auc": binary_auc(labels, predictions),
        "ndcgAt5": ndcg_at_k(rows, predictions, 5),
        "ndcgAt10": ndcg_at_k(rows, predictions, 10),
        "mrr": mean_reciprocal_rank(rows, predictions),
        "positiveRate": sum(1 for label in labels if label > 0) / len(labels),
    }


# Tái tạo Standard Ranking hiện tại từ đúng 8 trọng số production và penalty cuối vector.
# Đây là baseline để biết model AI có thực sự tốt hơn công thức deterministic hay chỉ học lại nó.
def standard_ranking_predictions(
    rows: list[RankingEvaluationRow],
) -> list[float]:
    """Sinh baseline deterministic để so sánh model trên cùng request groups."""

    predictions: list[float] = []
    for row in rows:
        # 8 feature đầu đóng góp điểm dương; negativePenalty là feature cuối nên bị trừ sau tổng.
        positive_score = sum(
            weight * value
            for weight, value in zip(STANDARD_RANKING_WEIGHTS, row.features[:-1], strict=True)
        )
        predictions.append(max(0.0, min(1.0, positive_score - row.features[-1])))
    return predictions


# Mô phỏng AI-Enhanced Ranking theo policy blend của Recommendation Service.
# Serving không thay Standard Ranking hoàn toàn: final = standard * (1 - blend) + AI * blend.
# Blend bị giới hạn tối đa 0.5 để model mới không thể lấn át hoàn toàn baseline an toàn.
def blended_ranking_predictions(
    standard_predictions: list[float],
    model_predictions: list[float],
    blend: float = 0.3,
) -> list[float]:
    """Trộn baseline và AI score để đo đúng semantics serving hiện tại."""

    # Hai danh sách phải cùng thứ tự candidate, nếu không phép blend sẽ ghép sai sản phẩm.
    if len(standard_predictions) != len(model_predictions):
        raise ValueError("METRIC_INPUT_LENGTH_MISMATCH")
    bounded_blend = max(0.0, min(0.5, blend))
    # Clamp final score để contract serving luôn trả score trong [0, 1].
    return [
        max(
            0.0,
            min(
                1.0,
                standard * (1.0 - bounded_blend) + model * bounded_blend,
            ),
        )
        for standard, model in zip(standard_predictions, model_predictions, strict=True)
    ]
