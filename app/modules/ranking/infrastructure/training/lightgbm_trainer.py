"""Huấn luyện LightGBM từ feature/label JSONL với split theo thời gian đơn giản và tái lập được."""

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _read_rows(path: Path, expected_features: int) -> tuple[list[list[float]], list[float]]:
    """Đọc dataset bounded, loại bỏ dòng malformed trước khi đưa vào trainer."""

    features: list[list[float]] = []
    labels: list[float] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row {line_number} must be an object")
            values = row.get("features")
            label = row.get("label")
            if (
                not isinstance(values, list)
                or len(values) != expected_features
                or not isinstance(label, (int, float))
                or not math.isfinite(float(label))
                or float(label) < 0
                or float(label) > 1
            ):
                raise ValueError(f"Invalid feature/label at line {line_number}")
            numeric = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"Invalid numeric feature at line {line_number}")
            features.append(numeric)
            labels.append(float(label))
    if len(features) < 20 or len(set(labels)) < 2:
        raise ValueError("Dataset must contain at least 20 rows and both labels")
    return features, labels


def train_model(input_path: Path, output_path: Path, version: str, expected_features: int) -> dict[str, Any]:
    """Train model offline, save artifact plus metadata để runtime load có kiểm soát."""

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LIGHTGBM_NOT_INSTALLED") from error

    features, labels = _read_rows(input_path, expected_features)
    split = max(10, min(len(features) - 10, int(len(features) * 0.8)))
    train_features, valid_features = features[:split], features[split:]
    train_labels, valid_labels = labels[:split], labels[split:]
    train_set = lgb.Dataset(train_features, label=train_labels, free_raw_data=False)
    valid_set = lgb.Dataset(valid_features, label=valid_labels, reference=train_set, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "data_random_seed": 42,
        },
        train_set,
        num_boost_round=200,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    if booster.num_feature() != expected_features:
        raise ValueError("TRAINED_MODEL_FEATURE_COUNT_MISMATCH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    metadata = {
        "modelVersion": version,
        "featureCount": expected_features,
        "trainRows": len(train_features),
        "validationRows": len(valid_features),
        "bestIteration": booster.best_iteration,
        "objective": "binary",
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    """CLI offline cho data scientist, không được import vào FastAPI request path."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train recommendation LightGBM ranking artifact")
    parser.add_argument("--input", required=True, type=Path, help="JSONL with features and label")
    parser.add_argument("--output", type=Path, default=Path(settings.ranking_model_path or "artifacts/ranking.txt"))
    parser.add_argument("--version", default=settings.ranking_model_version)
    parser.add_argument("--features", type=int, default=settings.ranking_expected_features)
    args = parser.parse_args(list(argv) if argv is not None else None)
    metadata = train_model(args.input, args.output, args.version, args.features)
    print(json.dumps(metadata, ensure_ascii=False))
