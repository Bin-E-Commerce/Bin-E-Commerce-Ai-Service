"""Đánh giá profileAffinity với baseline content-profile cosine trên dữ liệu synthetic.

Module chỉ đọc snapshot do simulator tạo và tính metric offline; không huấn luyện,
không kết nối database/Kafka, và không được dùng kết quả synthetic làm bằng chứng khách hàng thật.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.modules.ranking.domain.models import RANKING_FEATURE_SCHEMA_VERSION
from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RANKING_FEATURE_NAMES,
    RankingEvaluationRow,
    evaluate_predictions,
    profile_affinity_baseline_predictions,
    query_group_diagnostics,
)

_PROFILE_SIGNAL_STRENGTHS = {
    "product": 5.0,
    "category": 7.0,
    "brand": 6.0,
}


# Đọc JSONL của simulator và dừng ở dòng lỗi để dữ liệu thiếu không bị bỏ qua âm thầm.
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc JSONL object và báo lỗi có tên file, số dòng khi schema sai."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"INVALID_JSONL_LINE:{path.name}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL_ROW_MUST_BE_OBJECT:{path.name}:{line_number}")
            records.append(value)
    return records


# Dựng vector sở thích giả đúng các dimension của simulator để hai phương pháp nhận cùng profile đầu vào.
def _build_synthetic_user_vector(
    persona: dict[str, Any],
    session: dict[str, Any],
) -> dict[str, float]:
    """Tạo vector product/category/brand từ persona và anchor của request synthetic."""

    vector: dict[str, float] = {}
    current_product_id = session.get("currentProductId")
    if isinstance(current_product_id, str) and current_product_id:
        vector[f"product:{current_product_id}"] = _PROFILE_SIGNAL_STRENGTHS["product"]

    for dimension in ("category", "brand"):
        keys = persona.get(f"{dimension}Ids")
        if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"INVALID_SYNTHETIC_{dimension.upper()}_PROFILE")
        for key in keys:
            vector[f"{dimension}:{key}"] = _PROFILE_SIGNAL_STRENGTHS[dimension]
    return vector


# Biểu diễn candidate bằng cùng ba loại thuộc tính mà tiêu chí profileAffinity đang dùng.
def _build_synthetic_item_vector(item: dict[str, Any]) -> dict[str, float]:
    """Tạo vector nhị phân cho product ID, category ID và brand ID của item."""

    product_id = item.get("productId")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("SYNTHETIC_ITEM_PRODUCT_ID_REQUIRED")
    vector = {f"product:{product_id}": 1.0}
    for dimension in ("category", "brand"):
        key = item.get(f"{dimension}Id")
        if key is not None:
            if not isinstance(key, str) or not key:
                raise ValueError(f"INVALID_SYNTHETIC_ITEM_{dimension.upper()}_ID")
            vector[f"{dimension}:{key}"] = 1.0
    return vector


# Tính cosine giữa vector profile và item; profile rỗng cho điểm 0 thay vì chia cho 0.
def _cosine_similarity(
    user_vector: dict[str, float],
    item_vector: dict[str, float],
) -> float:
    """Trả cosine similarity hữu hạn trong [0, 1] cho các vector không âm."""

    values = [*user_vector.values(), *item_vector.values()]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("COSINE_VECTOR_VALUES_MUST_BE_FINITE_AND_NON_NEGATIVE")
    user_norm = math.sqrt(sum(value * value for value in user_vector.values()))
    item_norm = math.sqrt(sum(value * value for value in item_vector.values()))
    if user_norm == 0 or item_norm == 0:
        return 0.0
    dot_product = sum(value * item_vector.get(term, 0.0) for term, value in user_vector.items())
    score = dot_product / (user_norm * item_norm)
    return min(1.0, max(0.0, score))


# So sánh công thức dự án với cosine trên cùng request, item, nhãn và synthetic profile snapshot.
# Hàm chỉ nhận catalogSource synthetic để persona ẩn không vô tình được dùng như hồ sơ khách hàng thật.
# Feature và nhãn đều lấy từ test split; profile được đọc từ persona/session mà simulator đã tạo trước request.
def evaluate_synthetic_profile_cosine(dataset_dir: Path) -> dict[str, Any]:
    """Trả metric test cho profileAffinity và content-profile cosine trên dataset synthetic."""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("catalogSource") != "synthetic_catalog":
        raise ValueError("SYNTHETIC_CATALOG_REQUIRED_FOR_COSINE_COMPARISON")
    if manifest.get("featureSchemaVersion") != RANKING_FEATURE_SCHEMA_VERSION:
        raise ValueError("RANKING_FEATURE_SCHEMA_VERSION_MISMATCH")

    catalog = {item.get("productId"): item for item in _read_jsonl(dataset_dir / "catalog.snapshot.jsonl")}
    personas = {persona.get("userId"): persona for persona in _read_jsonl(dataset_dir / "personas.jsonl")}
    sessions = {session.get("requestId"): session for session in _read_jsonl(dataset_dir / "sessions.jsonl")}

    scored_rows: list[tuple[str, str, RankingEvaluationRow, float]] = []
    for value in _read_jsonl(dataset_dir / "test.jsonl"):
        features = value.get("features")
        label = value.get("label")
        request_id = value.get("requestId")
        user_id = value.get("userId")
        item_id = value.get("itemId")
        request_at = value.get("requestAt")
        if (
            not isinstance(features, list)
            or len(features) != len(RANKING_FEATURE_NAMES)
            or not all(isinstance(feature, (int, float)) and math.isfinite(feature) for feature in features)
            or isinstance(label, bool)
            or label not in (0, 1)
            or not isinstance(request_id, str)
            or not isinstance(user_id, str)
            or not isinstance(item_id, str)
            or not isinstance(request_at, str)
        ):
            raise ValueError("INVALID_SYNTHETIC_TEST_ROW")

        persona = personas.get(user_id)
        session = sessions.get(request_id)
        item = catalog.get(item_id)
        if persona is None or session is None or item is None:
            raise ValueError("SYNTHETIC_PROFILE_REFERENCE_MISSING")
        if session.get("userId") != user_id:
            raise ValueError("SYNTHETIC_REQUEST_USER_MISMATCH")
        try:
            request_time = datetime.fromisoformat(request_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("INVALID_SYNTHETIC_REQUEST_TIME") from error
        if request_time.tzinfo is None:
            raise ValueError("SYNTHETIC_REQUEST_TIMEZONE_REQUIRED")

        user_vector = _build_synthetic_user_vector(persona, session)
        item_vector = _build_synthetic_item_vector(item)
        scored_rows.append(
            (
                request_id,
                item_id,
                RankingEvaluationRow(
                    features=[float(feature) for feature in features],
                    label=float(label),
                    request_id=request_id,
                    request_time=request_time,
                ),
                _cosine_similarity(user_vector, item_vector),
            )
        )

    if not scored_rows:
        raise ValueError("SYNTHETIC_TEST_SPLIT_EMPTY")

    # Sort tie-break key independent of simulator impression order so equal scores do not inherit position bias.
    scored_rows.sort(key=lambda row: (row[0], row[1]))
    rows = [row[2] for row in scored_rows]
    cosine_scores = [row[3] for row in scored_rows]
    profile_scores = profile_affinity_baseline_predictions(rows)
    request_ids = {row.request_id for row in rows}
    profile_metrics = evaluate_predictions(rows, profile_scores)
    cosine_metrics = evaluate_predictions(rows, cosine_scores)
    # Delta dương nghĩa cosine cao hơn công thức dự án; metric không xác định thì giữ None.
    delta_metrics = {
        metric_name: (
            cosine_metrics[metric_name] - profile_metrics[metric_name]
            if cosine_metrics[metric_name] is not None and profile_metrics[metric_name] is not None
            else None
        )
        for metric_name in ("ndcgAt10", "recallAt10")
    }
    return {
        "evaluationType": "synthetic_only",
        "datasetVersion": manifest.get("datasetVersion"),
        "featureSchemaVersion": manifest.get("featureSchemaVersion"),
        "profileMethod": "content_profile_cosine",
        "baselineDefinition": {
            "userVector": "persona/session profile with product=5, category=7, brand=6",
            "itemVector": "binary productId/categoryId/brandId attributes",
            "score": "dot(user,item)/(norm(user)*norm(item))",
            "researchBasis": "Pazzani and Billsus (2007), Content-Based Recommendation Systems",
        },
        "profileSignalStrengths": _PROFILE_SIGNAL_STRENGTHS,
        "sameTestRowsAndLabels": True,
        "tieBreak": "ascending itemId, applied before ranking to both methods",
        "testRows": len(rows),
        "testRequests": len(request_ids),
        "queryDiagnostics": query_group_diagnostics(rows),
        "deltaContentProfileCosineMinusProjectProfileAffinity": delta_metrics,
        "metrics": {
            "projectProfileAffinity": profile_metrics,
            "contentProfileCosine": cosine_metrics,
        },
        "interpretationLimit": (
            "Persona/session profiles and outcomes are synthetic; this comparison does not establish real-user effectiveness."
        ),
    }
