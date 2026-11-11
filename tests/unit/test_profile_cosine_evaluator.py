"""Kiểm thử baseline cosine synthetic không cần LightGBM, database hoặc Kafka."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.modules.ranking.infrastructure.training.profile_cosine_evaluator import (
    evaluate_synthetic_profile_cosine,
)


# Ghi fixture JSONL nhỏ để public evaluator chạy cùng contract với dataset simulator.
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Tạo file JSONL fixture độc lập, không chạm dataset thật."""

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


# Dựng hai sản phẩm để baseline cosine xếp item khớp profile trước item không liên quan.
def _write_dataset(dataset_dir: Path, catalog_source: str = "synthetic_catalog") -> None:
    """Tạo fixture gồm persona, request, catalog và test rows trong cùng một nhóm."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "datasetVersion": "ranking-synthetic-v1",
                "featureSchemaVersion": "ranking-features-v2",
                "catalogSource": catalog_source,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset_dir / "catalog.snapshot.jsonl",
        [
            {"productId": "item-match", "categoryId": "cat-liked", "brandId": "brand-liked"},
            {"productId": "item-other", "categoryId": "cat-other", "brandId": "brand-other"},
        ],
    )
    _write_jsonl(
        dataset_dir / "personas.jsonl",
        [{"userId": "user-1", "categoryIds": ["cat-liked"], "brandIds": ["brand-liked"]}],
    )
    _write_jsonl(
        dataset_dir / "sessions.jsonl",
        [{"userId": "user-1", "requestId": "request-1", "currentProductId": None}],
    )
    _write_jsonl(
        dataset_dir / "test.jsonl",
        [
            {
                "features": [0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                "label": 1,
                "requestId": "request-1",
                "requestAt": "2026-11-01T00:00:00Z",
                "userId": "user-1",
                "itemId": "item-match",
            },
            {
                "features": [0.8, 0, 0, 0, 0, 0, 0, 0, 0],
                "label": 0,
                "requestId": "request-1",
                "requestAt": "2026-11-01T00:00:00Z",
                "userId": "user-1",
                "itemId": "item-other",
            },
        ],
    )


# Baseline profile cosine doit suivre le même request group et retrouver l'item qui matche le profil.
def test_evaluate_synthetic_profile_cosine_compares_same_candidates(tmp_path: Path) -> None:
    """Les deux scoreurs reçoivent les mêmes labels et le cosine classe le match en premier."""

    _write_dataset(tmp_path)

    result = evaluate_synthetic_profile_cosine(tmp_path)

    assert result["testRows"] == 2
    assert result["testRequests"] == 1
    assert result["metrics"]["contentProfileCosine"]["ndcgAt10"] == pytest.approx(1.0)
    assert result["metrics"]["projectProfileAffinity"]["ndcgAt10"] == pytest.approx(1 / 1.584962500721156)
    assert result["deltaContentProfileCosineMinusProjectProfileAffinity"]["ndcgAt10"] == pytest.approx(1 - 1 / 1.584962500721156)


# Hoán vị file candidate không được làm đổi kết quả do test input giữ thứ tự display rank.
def test_evaluate_synthetic_profile_cosine_ignores_input_candidate_order(
    tmp_path: Path,
) -> None:
    """Tie-break theo product ID giữ metric ổn định khi JSONL đổi thứ tự dòng."""

    _write_dataset(tmp_path)
    original = evaluate_synthetic_profile_cosine(tmp_path)
    test_path = tmp_path / "test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(test_path, list(reversed(rows)))

    permuted = evaluate_synthetic_profile_cosine(tmp_path)

    assert permuted["metrics"] == original["metrics"]


# Chặn dùng persona giả để đánh giá impression hoặc hồ sơ của người dùng thật.
def test_evaluate_synthetic_profile_cosine_rejects_non_synthetic_catalog(
    tmp_path: Path,
) -> None:
    """Evaluator không giả vờ hỗ trợ dữ liệu thật khi đầu vào không có profile contract tương ứng."""

    _write_dataset(tmp_path, catalog_source="recommendation_catalog_products")

    with pytest.raises(ValueError, match="SYNTHETIC_CATALOG_REQUIRED_FOR_COSINE_COMPARISON"):
        evaluate_synthetic_profile_cosine(tmp_path)
