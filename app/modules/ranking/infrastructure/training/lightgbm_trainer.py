"""Huấn luyện artifact LightGBM từ dataset JSONL ở ngoài request path.

Module này chỉ thuộc infrastructure: đọc dữ liệu offline, kiểm tra contract
feature, train model và ghi artifact/metadata. Nó không được import vào FastAPI
request path, không gọi Kafka và không tạo dữ liệu nghiệp vụ.
"""

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RankingEvaluationRow,
    blended_ranking_predictions,
    evaluate_predictions,
    standard_ranking_predictions,
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
                or float(label) < 0
                or float(label) > 1
            ):
                raise ValueError(f"INVALID_FEATURE_OR_LABEL:{line_number}")
            # Chuyển mọi giá trị số về float và từ chối NaN/Infinity hoặc feature ngoài [0, 1].
            try:
                numeric = [float(item) for item in values]
            except (TypeError, ValueError) as error:
                raise ValueError(f"INVALID_FEATURE_VALUE:{line_number}") from error
            if not all(
                math.isfinite(item) and 0 <= item <= 1 for item in numeric
            ):
                raise ValueError(f"FEATURE_OUT_OF_RANGE:{line_number}")
            # requestId được giữ lại để evaluator đo NDCG/MRR theo từng recommendation request.
            request_id = value.get("requestId")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError(f"INVALID_REQUEST_ID:{line_number}")
            rows.append(
                RankingEvaluationRow(
                    features=numeric,
                    label=float(label),
                    request_id=request_id,
                )
            )
    return rows


# Kiểm tra điều kiện tối thiểu của từng split trước khi gọi LightGBM hoặc metric.
# Train cần đủ dòng để model học; validation/test cần label 0 và 1 để metric không bị vô nghĩa.
# Đây là lớp bảo vệ sớm, giúp lỗi dataset xuất hiện trước khi tốn CPU train model.
def _validate_split(
    rows: list[RankingEvaluationRow],
    name: str,
    minimum_rows: int,
    require_both_labels: bool = True,
) -> None:
    """Đảm bảo split đủ lớn và không bị rỗng class theo contract dataset."""

    # Split quá nhỏ thường cho metric dao động mạnh và có thể làm LightGBM không train được.
    if len(rows) < minimum_rows:
        raise ValueError(f"{name.upper()}_ROWS_TOO_FEW:{len(rows)}")
    # Binary ranking phải có cả candidate positive và negative để học thứ tự phân biệt.
    labels = {row.label for row in rows}
    if require_both_labels and len(labels) < 2:
        raise ValueError(f"{name.upper()}_MUST_CONTAIN_BOTH_LABELS")


# Chuyển row domain thành ma trận và vector label mà LightGBM yêu cầu.
# Import numpy bên trong hàm để evaluator/parser nhẹ hơn và để lỗi dependency xuất hiện
# đúng lúc chạy training, không làm hỏng các route AI chỉ cần load model đã có.
def _to_arrays(rows: list[RankingEvaluationRow]) -> tuple[Any, Any]:
    """Tạo ndarray lazy để module parser/evaluator vẫn không cần numpy."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("LIGHTGBM_OR_NUMPY_NOT_INSTALLED") from error
    # LightGBM cần ma trận số và vector label; float32 giảm bộ nhớ cho dataset lớn.
    return (
        np.asarray([row.features for row in rows], dtype=np.float32),
        np.asarray([row.label for row in rows], dtype=np.float32),
    )


# Tính metadata metric cho một split sau khi artifact đã sinh prediction.
# Model được gọi đúng một lần cho split, sau đó cùng prediction được dùng cho LogLoss/AUC/NDCG/MRR
# để mọi metric đang mô tả cùng một kết quả model.
def _evaluate_split(
    booster: Any,
    rows: list[RankingEvaluationRow],
) -> dict[str, float | None]:
    """Dùng đúng prediction của booster để tính binary và ranking metrics."""

    features, _ = _to_arrays(rows)
    predictions = [float(value) for value in booster.predict(features)]
    return evaluate_predictions(rows, predictions)


# Tính cả baseline và AI-enhanced score để report không chỉ chứng minh model train được.
# `standardRanking` là công thức deterministic hiện tại, `aiRanking` là output LightGBM thuần,
# còn `aiEnhancedRanking` mô phỏng đúng score blend khi Recommendation Service phục vụ request.
def _evaluate_comparison(
    booster: Any,
    rows: list[RankingEvaluationRow],
) -> dict[str, dict[str, float | None]]:
    """So sánh Standard/Hybrid deterministic với AI và score blend."""

    features, _ = _to_arrays(rows)
    model_predictions = [float(value) for value in booster.predict(features)]
    standard_predictions = standard_ranking_predictions(rows)
    enhanced_predictions = blended_ranking_predictions(
        standard_predictions,
        model_predictions,
    )
    return {
        "standardRanking": evaluate_predictions(rows, standard_predictions),
        "aiRanking": evaluate_predictions(rows, model_predictions),
        "aiEnhancedRanking": evaluate_predictions(rows, enhanced_predictions),
    }


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

    # LightGBM và numpy chỉ cần cho offline job; FastAPI runtime không gọi hàm train này.
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LIGHTGBM_OR_NUMPY_NOT_INSTALLED") from error

    # Hai version này đi vào metadata artifact để biết model được train từ dataset/schema nào.
    # Dataset synthetic phải có đủ ba split và manifest; trainer không tự tạo split mới.
    train_rows = _read_rows(dataset_dir / "train.jsonl", expected_features)
    validation_rows = _read_rows(
        dataset_dir / "validation.jsonl", expected_features
    )
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
    if (
        not isinstance(feature_schema_version_value, str)
        or not feature_schema_version_value
    ):
        raise ValueError("FEATURE_SCHEMA_VERSION_REQUIRED_IN_MANIFEST")
    # Version lấy từ manifest để metadata phản ánh đúng dataset đã dùng, không dùng tên demo hard-code.
    dataset_version = dataset_version_value
    feature_schema_version = feature_schema_version_value
    seed = int(manifest.get("seed", seed))
    # Ba split tự kiểm tra label trước khi train; test bắt buộc tồn tại để không quay lại mô hình 80/20.
    _validate_split(train_rows, "train", 20)
    _validate_split(validation_rows, "validation", 1)
    _validate_split(test_rows, "test", 1)

    # Chuyển ba phần dữ liệu sang ndarray một lần để tránh chuyển đổi lặp lại trong LightGBM.
    train_features, train_labels = _to_arrays(train_rows)
    validation_features, validation_labels = _to_arrays(validation_rows)
    # train_set là dữ liệu model được phép học; validation_set chỉ dùng theo dõi và early stopping.
    train_set = lgb.Dataset(
        train_features,
        label=train_labels,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_features,
        label=validation_labels,
        reference=train_set,
        free_raw_data=False,
    )
    # Binary objective phù hợp label 0/1 của V1. Seed cố định giúp cùng dataset cho kết quả tái lập.
    # early_stopping dừng khi validation không tốt thêm, hạn chế học quá mức trên dữ liệu synthetic.
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 3,
            "min_sum_hessian_in_leaf": 1e-3,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        },
        train_set,
        num_boost_round=200,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    # Kiểm tra artifact ngay sau train để không lưu model có schema khác với serving.
    if booster.num_feature() != expected_features:
        raise ValueError("TRAINED_MODEL_FEATURE_COUNT_MISMATCH")

    # Artifact, metadata và metrics dùng tên output riêng để mỗi lần train có thể truy xuất độc lập.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    validation_metrics = _evaluate_split(booster, validation_rows)
    test_metrics = _evaluate_split(booster, test_rows)
    validation_comparison = _evaluate_comparison(booster, validation_rows)
    test_comparison = _evaluate_comparison(booster, test_rows)
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
        "positiveRate": sum(row.label for row in train_rows) / len(train_rows),
        "bestIteration": booster.best_iteration,
        "objective": "binary",
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
    parser = argparse.ArgumentParser(
        description="Train recommendation LightGBM ranking artifact"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl, validation.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            settings.ranking_model_path
            or "artifacts/synthetic/ranking-lgbm-synthetic-v1.txt"
        ),
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
