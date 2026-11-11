"""Huấn luyện artifact LightGBM từ dataset JSONL ngoài request path.

Module này chịu trách nhiệm kiểm tra dataset, train LambdaRank và ghi metadata.
Nó không được import vào FastAPI request path, không gọi Kafka và không tạo dữ liệu nghiệp vụ.
"""

import argparse
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.modules.ranking.domain.models import (
    RANKING_FEATURE_MONOTONE_DIRECTIONS,
    RANKING_FEATURE_SCHEMA_VERSION,
)
from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RANKING_FEATURE_NAMES,
    RankingEvaluationRow,
    evaluate_predictions,
    feature_only_baseline_predictions,
    popularity_baseline_predictions,
    profile_affinity_baseline_predictions,
    query_group_diagnostics,
)


# Đọc và kiểm tra từng dòng JSONL trước khi chuyển dataset sang LightGBM.
# Simulator ghi mỗi candidate thành một dòng độc lập để có thể stream file lớn,
# nhưng trainer vẫn phải kiểm tra từng dòng vì một dòng hỏng không được âm thầm đi vào model.
# Hàm chỉ đọc và trả về row in-memory; không sửa file nguồn, không gọi database/Kafka.
def _read_rows(path: Path, expected_features: int) -> list[RankingEvaluationRow]:
    """Đọc feature, label và request group; từ chối dòng sai contract ngay lập tức."""

    # Giữ request_id để evaluator có thể tính NDCG/MRR theo từng recommendation request.
    rows: list[RankingEvaluationRow] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            # Parse từng dòng thay vì json array lớn để dataset có thể được sinh/import dạng JSONL.
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"INVALID_JSONL_LINE:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"DATASET_ROW_MUST_BE_OBJECT:{line_number}")
            # Feature count phải khớp artifact; sai số cột sẽ làm model train và serving lệch schema.
            values = value.get("features")
            label = value.get("label")
            if (
                not isinstance(values, list)
                or len(values) != expected_features
                or isinstance(label, bool)
                or not isinstance(label, (int, float))
                or not math.isfinite(float(label))
                or float(label) not in (0.0, 1.0)
            ):
                raise ValueError(f"INVALID_FEATURE_OR_LABEL:{line_number}")
            # Chuyển mọi giá trị số về float và từ chối NaN/Infinity hoặc feature ngoài [0, 1].
            try:
                numeric = [float(item) for item in values]
            except (TypeError, ValueError) as error:
                raise ValueError(f"INVALID_FEATURE_VALUE:{line_number}") from error
            if not all(math.isfinite(item) and 0 <= item <= 1 for item in numeric):
                raise ValueError(f"FEATURE_OUT_OF_RANGE:{line_number}")
            # LambdaRank cần requestId để phân nhóm candidate; trainer sẽ từ chối row không có ID.
            request_id = value.get("requestId")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError(f"INVALID_REQUEST_ID:{line_number}")
            request_at = value.get("requestAt")
            if not isinstance(request_at, str):
                raise ValueError(f"REQUEST_TIME_REQUIRED:{line_number}")
            try:
                parsed_request_time = datetime.fromisoformat(request_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"INVALID_REQUEST_TIME:{line_number}") from error
            if parsed_request_time.tzinfo is None:
                raise ValueError(f"REQUEST_TIMEZONE_REQUIRED:{line_number}")
            rows.append(
                RankingEvaluationRow(
                    features=numeric,
                    label=float(label),
                    request_id=request_id,
                    request_time=parsed_request_time.astimezone(UTC),
                )
            )
    return rows


# Kiểm tra điều kiện tối thiểu của từng split trước khi gọi LightGBM hoặc metric.
# Mỗi split cần label 0/1 và ít nhất một request có cả hai để tạo phép so sánh thứ tự.
# Đây là lớp bảo vệ sớm, giúp lỗi dataset xuất hiện trước khi tốn CPU train model.
def _validate_split(
    rows: list[RankingEvaluationRow],
    name: str,
    minimum_rows: int,
    require_both_labels: bool = True,
) -> None:
    """Đảm bảo split đủ lớn và có query group hữu ích cho LambdaRank."""

    # Split quá nhỏ thường cho metric dao động mạnh và có thể làm LightGBM không train được.
    if len(rows) < minimum_rows:
        raise ValueError(f"{name.upper()}_ROWS_TOO_FEW:{len(rows)}")
    # Dataset vẫn cần cả candidate positive và negative, nhưng chúng phải cùng một request.
    labels = {row.label for row in rows}
    if require_both_labels and len(labels) < 2:
        raise ValueError(f"{name.upper()}_MUST_CONTAIN_BOTH_LABELS")
    # Ranking loss cần ít nhất một request có positive lẫn negative để tạo cặp so sánh.
    grouped: dict[str, set[float]] = {}
    for row in rows:
        if not row.request_id:
            raise ValueError(f"{name.upper()}_REQUEST_ID_REQUIRED")
        grouped.setdefault(row.request_id, set()).add(row.label)
    if not any(len(group_labels) > 1 for group_labels in grouped.values()):
        raise ValueError(f"{name.upper()}_NO_INFORMATIVE_RANKING_GROUP")


# Chặn dataset cũ có cùng số cột nhưng feature semantics khác với code đang phục vụ.
def _validate_feature_schema_version(
    value: object,
    expected: str,
) -> str:
    """Chỉ nhận manifest mang đúng version schema do service khai báo."""

    if not isinstance(value, str) or value != expected:
        raise ValueError("FEATURE_SCHEMA_VERSION_MISMATCH")
    return value


# Chứng minh train, validation và test đi theo thời gian và không chia cùng request qua hai tập.
def _validate_temporal_order(
    train_rows: list[RankingEvaluationRow],
    validation_rows: list[RankingEvaluationRow],
    test_rows: list[RankingEvaluationRow],
) -> None:
    """Từ chối split thiếu thời gian, trùng request hoặc sai thứ tự thời gian."""

    split_groups = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    request_sets: dict[str, set[str]] = {}
    time_ranges: dict[str, tuple[datetime, datetime]] = {}
    for split_name, rows in split_groups.items():
        request_times: dict[str, datetime] = {}
        for row in rows:
            if not row.request_id or row.request_time is None:
                raise ValueError("TEMPORAL_SPLIT_REQUIRES_REQUEST_ID_AND_TIME")
            existing_time = request_times.get(row.request_id)
            if existing_time is not None and existing_time != row.request_time:
                raise ValueError("REQUEST_TIME_MUST_MATCH_WITHIN_QUERY")
            request_times[row.request_id] = row.request_time
        request_sets[split_name] = set(request_times)
        timestamps = list(request_times.values())
        if not timestamps:
            raise ValueError("TEMPORAL_SPLIT_CANNOT_BE_EMPTY")
        time_ranges[split_name] = (min(timestamps), max(timestamps))
    if (
        request_sets["train"] & request_sets["validation"]
        or request_sets["train"] & request_sets["test"]
        or request_sets["validation"] & request_sets["test"]
    ):
        raise ValueError("REQUEST_GROUP_LEAKAGE_ACROSS_SPLITS")
    if not (time_ranges["train"][1] < time_ranges["validation"][0] and time_ranges["validation"][1] < time_ranges["test"][0]):
        raise ValueError("SPLITS_MUST_BE_STRICTLY_CHRONOLOGICAL")


# Sắp xếp các candidate thành query group liên tục theo định dạng LightGBM LambdaRank.
def _group_for_ranking(
    rows: list[RankingEvaluationRow],
) -> tuple[list[RankingEvaluationRow], list[int]]:
    """Sắp rows liền nhau theo requestId và trả kích thước query cho LightGBM."""

    # LightGBM group dùng kích thước liên tiếp, vì vậy map theo lần xuất hiện giữ order tái lập.
    groups: dict[str, list[RankingEvaluationRow]] = {}
    for row in rows:
        if not row.request_id:
            raise ValueError("RANKING_REQUEST_ID_REQUIRED")
        groups.setdefault(row.request_id, []).append(row)
    # Query một candidate không tạo cặp ranking; loại khỏi fit thay vì giả định relevance của item khác.
    usable = [group for group in groups.values() if len(group) >= 2]
    if not usable:
        raise ValueError("RANKING_GROUPS_REQUIRE_AT_LEAST_TWO_ITEMS")
    return [row for group in usable for row in group], [len(group) for group in usable]


# Chuyển rows có thể chọn cột ablation thành ndarray và label nhị phân cho LightGBM.
def _to_arrays(rows: list[RankingEvaluationRow], feature_indices: tuple[int, ...] | None = None) -> tuple[Any, Any]:
    """Tạo ndarray lazy; numpy chỉ cần có trong tiến trình training."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("LIGHTGBM_OR_NUMPY_NOT_INSTALLED") from error
    # LightGBM cần ma trận số và vector label; float32 giảm bộ nhớ cho dataset lớn.
    columns = feature_indices or tuple(range(len(rows[0].features)))
    # Chọn cùng feature subset cho train/validation/test để ablation không đổi thứ tự cột.
    return (
        np.asarray(
            [[row.features[index] for index in columns] for row in rows],
            dtype=np.float32,
        ),
        np.asarray([row.label for row in rows], dtype=np.float32),
    )


# Affinity chỉ được giữ nguyên hoặc tăng dự đoán khi feature tăng; penalty âm chỉ được giữ nguyên hoặc giảm.
# Ràng buộc được ánh xạ theo tên để ablation bỏ feature không làm lệch cột còn lại.
def _monotone_constraints(
    feature_indices: tuple[int, ...] | None = None,
) -> list[int]:
    """Trả hướng đơn điệu cho đúng các cột feature còn lại trong model."""

    # Khi feature bị bỏ khỏi ablation, lấy hướng theo tên cột gốc để không gán nhầm constraint.
    selected = feature_indices or tuple(range(len(RANKING_FEATURE_NAMES)))
    return [RANKING_FEATURE_MONOTONE_DIRECTIONS.get(RANKING_FEATURE_NAMES[index], 0) for index in selected]


# Tính metric ranking trên đúng một lần dự đoán raw score cho split.
def _evaluate_split(
    booster: Any,
    rows: list[RankingEvaluationRow],
    feature_indices: tuple[int, ...] | None = None,
) -> dict[str, float | None]:
    """Dùng đúng prediction của booster để tính AUC và metric top-K."""

    features, _ = _to_arrays(rows, feature_indices)
    predictions = [float(value) for value in booster.predict(features)]
    return evaluate_predictions(rows, predictions)


# Train một model riêng cho ablation; test set không được dùng để chọn số vòng lặp.
def _train_ranker(
    train_rows: list[RankingEvaluationRow],
    validation_rows: list[RankingEvaluationRow],
    *,
    seed: int,
    feature_indices: tuple[int, ...] | None = None,
) -> Any:
    """Train LambdaRank theo request group, dùng validation NDCG@10 để dừng sớm."""

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LIGHTGBM_OR_NUMPY_NOT_INSTALLED") from error
    ordered_train, train_groups = _group_for_ranking(train_rows)
    ordered_validation, validation_groups = _group_for_ranking(validation_rows)
    train_features, train_labels = _to_arrays(ordered_train, feature_indices)
    validation_features, validation_labels = _to_arrays(ordered_validation, feature_indices)
    train_set = lgb.Dataset(
        train_features,
        label=train_labels,
        group=train_groups,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_features,
        label=validation_labels,
        group=validation_groups,
        reference=train_set,
        free_raw_data=False,
    )
    # LambdaRank tối ưu thứ tự theo query; label_gain nhị phân khớp click/không click.
    return lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "monotone_constraints": _monotone_constraints(feature_indices),
            "ndcg_eval_at": [10, 5],
            "label_gain": [0, 1],
            "lambdarank_truncation_level": 10,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        },
        train_set,
        num_boost_round=300,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(
                20,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )


# Tính baseline chỉ để đánh giá; endpoint serving không gọi các phép xếp thứ tự này.
def _evaluate_comparison(
    booster: Any,
    ablated_boosters: dict[str, tuple[Any, tuple[int, ...]]],
    rows: list[RankingEvaluationRow],
) -> dict[str, object]:
    """So sánh full model, hai baseline và model được train lại khi bỏ từng feature."""

    features, _ = _to_arrays(rows)
    model_predictions = [float(value) for value in booster.predict(features)]
    profile_predictions = profile_affinity_baseline_predictions(rows)
    popular_predictions = popularity_baseline_predictions(rows)
    feature_only_metrics = {
        feature_name: evaluate_predictions(
            rows,
            feature_only_baseline_predictions(rows, feature_name),
        )
        for feature_name in RANKING_FEATURE_NAMES
    }
    ablation_metrics = {
        feature_name: evaluate_predictions(
            rows,
            _predict_with_features(ablated_booster, rows, feature_indices),
        )
        for feature_name, (ablated_booster, feature_indices) in ablated_boosters.items()
    }
    return {
        "lambdaRank": evaluate_predictions(rows, model_predictions),
        "profileAffinityOnly": evaluate_predictions(rows, profile_predictions),
        "popularityOnly": evaluate_predictions(rows, popular_predictions),
        "featureOnly": feature_only_metrics,
        "withoutFeature": ablation_metrics,
    }


# Dự đoán trên feature subset để mô hình ablation dùng đúng 8 cột đã train.
def _predict_with_features(
    booster: Any,
    rows: list[RankingEvaluationRow],
    feature_indices: tuple[int, ...],
) -> list[float]:
    """Trả raw score của booster ablation theo thứ tự row ban đầu."""

    features, _ = _to_arrays(rows, feature_indices)
    return [float(value) for value in booster.predict(features)]


# Train model từ ba split explicit của simulator.
# Simulator đã chia train/validation/test theo session nên trainer không được chia lại ngẫu nhiên.
# Quy tắc này giữ ranh giới request, tránh data leakage và bảo đảm test là dữ liệu chưa dùng để train.
def train_model(
    output_path: Path,
    version: str,
    expected_features: int,
    *,
    dataset_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Train artifact từ dataset synthetic đã chia sẵn và ghi metadata/metrics cạnh model."""

    # Dataset phải có ba split và manifest; trainer không chia lại ngẫu nhiên để tránh leakage.
    train_rows = _read_rows(dataset_dir / "train.jsonl", expected_features)
    validation_rows = _read_rows(dataset_dir / "validation.jsonl", expected_features)
    test_rows = _read_rows(dataset_dir / "test.jsonl", expected_features)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("DATASET_MANIFEST_REQUIRED")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("DATASET_MANIFEST_MUST_BE_OBJECT")
    dataset_version_value = manifest.get("datasetVersion")
    feature_schema_version_value = manifest.get("featureSchemaVersion")
    if not isinstance(dataset_version_value, str) or not dataset_version_value:
        raise ValueError("DATASET_VERSION_REQUIRED_IN_MANIFEST")
    # Chỉ train dataset đã sinh từ đúng semantics feature hiện hành; cùng số cột chưa đủ an toàn.
    feature_schema_version = _validate_feature_schema_version(
        feature_schema_version_value,
        RANKING_FEATURE_SCHEMA_VERSION,
    )
    # Version lấy từ manifest để metadata phản ánh đúng dataset đã dùng, không dùng tên demo hard-code.
    dataset_version = dataset_version_value
    seed = int(manifest.get("seed", seed))
    # Ba split kiểm tra group/label trước khi train; test tuyệt đối không tham gia fit hay early stopping.
    _validate_split(train_rows, "train", 20)
    _validate_split(validation_rows, "validation", 1)
    _validate_split(test_rows, "test", 1)
    _validate_temporal_order(train_rows, validation_rows, test_rows)
    # Model đầy đủ học từ mọi tín hiệu; mỗi ranker ablation bỏ đúng một feature.
    if expected_features != len(RANKING_FEATURE_NAMES):
        raise ValueError("RANKING_FEATURE_SCHEMA_VERSION_MISMATCH")
    booster = _train_ranker(train_rows, validation_rows, seed=seed)
    # Huấn luyện lại cùng split train/validation; chỉ đo test sau khi đã cố định mọi model.
    ablated_boosters: dict[str, tuple[Any, tuple[int, ...]]] = {}
    for omitted_index, feature_name in enumerate(RANKING_FEATURE_NAMES):
        selected_indices = tuple(index for index in range(expected_features) if index != omitted_index)
        ablated_boosters[feature_name] = (
            _train_ranker(
                train_rows,
                validation_rows,
                seed=seed,
                feature_indices=selected_indices,
            ),
            selected_indices,
        )
    # Kiểm tra artifact ngay sau train để không lưu model có schema khác với serving.
    if booster.num_feature() != expected_features:
        raise ValueError("TRAINED_MODEL_FEATURE_COUNT_MISMATCH")

    # Artifact, metadata và metrics dùng tên output riêng để mỗi lần train có thể truy xuất độc lập.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    validation_metrics = _evaluate_split(booster, validation_rows)
    test_metrics = _evaluate_split(booster, test_rows)
    validation_comparison = _evaluate_comparison(
        booster,
        ablated_boosters,
        validation_rows,
    )
    test_comparison = _evaluate_comparison(
        booster,
        ablated_boosters,
        test_rows,
    )
    # Metadata là phần truy xuất nguồn gốc của model: version, seed, số dòng và chất lượng từng split.
    metadata: dict[str, Any] = {
        "modelVersion": version,
        "featureCount": expected_features,
        "featureSchemaVersion": feature_schema_version,
        "datasetVersion": dataset_version,
        "seed": seed,
        "trainRows": len(train_rows),
        "validationRows": len(validation_rows),
        "testRows": len(test_rows),
        "trainQueryGroups": len(_group_for_ranking(train_rows)[1]),
        "validationQueryGroups": len(_group_for_ranking(validation_rows)[1]),
        "testQueryGroups": len(_group_for_ranking(test_rows)[1]),
        "queryDiagnostics": {
            "train": query_group_diagnostics(train_rows),
            "validation": query_group_diagnostics(validation_rows),
            "test": query_group_diagnostics(test_rows),
        },
        "positiveRate": sum(row.label for row in train_rows) / len(train_rows),
        "bestIteration": booster.best_iteration,
        "objective": "lambdarank",
        "monotoneConstraints": _monotone_constraints(),
        "rankingMetric": "NDCG@10",
        "featureNames": list(RANKING_FEATURE_NAMES),
        "featureAblation": "Each feature omitted in a separately retrained model; compare on identical held-out queries.",
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
        "comparisonMetrics": {
            "validation": validation_comparison,
            "test": test_comparison,
        },
    }
    # File meta đi cạnh artifact để registry/operator kiểm tra model mà không đọc lại dataset lớn.
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_model_metrics(output_path, metadata)
    return metadata


# Ghi model metric cạnh artifact để dataset input có thể được mount read-only.
# Không ghi ngược vào metrics.json của dataset vì dataset là input bất biến và thường được mount :ro.
def _write_model_metrics(output_path: Path, metadata: dict[str, Any]) -> None:
    """Ghi sidecar metric model mà không mutate thư mục dataset."""

    # Sidecar này chứa riêng metric model và comparison để đọc nhanh sau mỗi lần train.
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics = {
        "modelVersion": metadata["modelVersion"],
        "datasetVersion": metadata["datasetVersion"],
        "validationMetrics": metadata["validationMetrics"],
        "testMetrics": metadata["testMetrics"],
        "comparisonMetrics": metadata["comparisonMetrics"],
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Parse CLI cho training dataset synthetic có split sẵn.
# CLI là entrypoint offline, không được gọi từ HTTP request; dataset directory là input duy nhất.
def main(argv: Iterable[str] | None = None) -> None:
    """Chạy training job độc lập với tiến trình FastAPI."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train recommendation LightGBM ranking artifact")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl, validation.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.ranking_model_path or "artifacts/synthetic/ranking-lambdarank-synthetic-v1.txt"),
    )
    parser.add_argument("--version", default=settings.ranking_model_version)
    parser.add_argument("--features", type=int, default=settings.ranking_expected_features)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    # In metadata dạng JSON để shell/CI có thể lưu lại kết quả run mà không cần parse log LightGBM.
    metadata = train_model(
        args.output,
        args.version,
        args.features,
        dataset_dir=args.dataset_dir,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False))
