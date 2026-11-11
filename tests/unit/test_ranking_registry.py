"""Kiểm thử registry từ chối artifact sai objective hoặc sai schema."""

import json
import sys
from types import SimpleNamespace

import pytest

from app.modules.ranking.infrastructure.registry.model_registry import (
    LightGbmRankingModel,
)


# Tạo sidecar hợp lệ để kiểm tra objective thật được đọc từ artifact model.
def _write_metadata(
    model_path,
    *,
    objective: str = "lambdarank",
    feature_schema_version: str = "ranking-features-v2",
    monotone_constraints: list[int] | None = None,
) -> None:
    """Ghi metadata tối thiểu theo contract do trainer tạo."""

    model_path.with_suffix(model_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "modelVersion": "ranking-test-v1",
                "featureCount": 9,
                "featureSchemaVersion": feature_schema_version,
                "objective": objective,
                "featureNames": [
                    "profileAffinity",
                    "sessionContext",
                    "semanticSimilarity",
                    "coBehavior",
                    "popularity",
                    "freshness",
                    "quality",
                    "exploration",
                    "negativePenalty",
                ],
                "monotoneConstraints": monotone_constraints or [1, 0, 0, 0, 0, 0, 0, 0, -1],
            }
        ),
        encoding="utf-8",
    )


# Sidecar không được quyền khai báo một binary classifier thành LambdaRank.
def test_registry_rejects_artifact_with_non_ranking_objective(
    tmp_path,
    monkeypatch,
) -> None:
    """Objective trong model LightGBM phải đúng với metadata sidecar."""

    model_path = tmp_path / "ranking.txt"
    _write_metadata(model_path)

    class FakeBooster:
        def __init__(self, model_file: str) -> None:
            self.model_file = model_file

        def dump_model(self) -> dict[str, str]:
            return {"objective": "binary sigmoid:1"}

        def num_feature(self) -> int:
            return 9

    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(Booster=FakeBooster),
    )

    with pytest.raises(ValueError, match="RANKING_MODEL_OBJECTIVE_MISMATCH"):
        LightGbmRankingModel(str(model_path), "ranking-test-v1")


# Feature count của artifact phải trùng sidecar để ngăn serving sai thứ tự/cỡ vector.
def test_registry_rejects_artifact_with_wrong_feature_count(
    tmp_path,
    monkeypatch,
) -> None:
    """Artifact có tám cột không được nạp khi contract yêu cầu chín feature."""

    model_path = tmp_path / "ranking.txt"
    _write_metadata(model_path)

    class FakeBooster:
        def __init__(self, model_file: str) -> None:
            self.model_file = model_file

        def dump_model(self) -> dict[str, str]:
            return {"objective": "lambdarank"}

        def num_feature(self) -> int:
            return 8

    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(Booster=FakeBooster),
    )

    with pytest.raises(ValueError, match="RANKING_MODEL_FEATURE_COUNT_MISMATCH"):
        LightGbmRankingModel(str(model_path), "ranking-test-v1")


# Cùng objective và số cột nhưng semantics feature v1 không được nạp theo hợp đồng v2.
def test_registry_rejects_artifact_with_old_feature_schema(tmp_path) -> None:
    """Metadata phải khớp chính xác với schema hiện hành trước khi load LightGBM."""

    model_path = tmp_path / "ranking.txt"
    _write_metadata(model_path, feature_schema_version="ranking-features-v1")

    with pytest.raises(ValueError, match="RANKING_MODEL_METADATA_MISMATCH"):
        LightGbmRankingModel(str(model_path), "ranking-test-v1")


# Model cũ không có chiều tăng được bảo đảm cho affinity nên không được dùng theo policy mới.
def test_registry_rejects_artifact_without_monotone_profile_affinity(tmp_path) -> None:
    """Artifact phải khai báo positive constraint trên profileAffinity."""

    model_path = tmp_path / "ranking.txt"
    _write_metadata(
        model_path,
        monotone_constraints=[0, 0, 0, 0, 0, 0, 0, 0, -1],
    )

    with pytest.raises(ValueError, match="RANKING_MODEL_METADATA_MISMATCH"):
        LightGbmRankingModel(str(model_path), "ranking-test-v1")
