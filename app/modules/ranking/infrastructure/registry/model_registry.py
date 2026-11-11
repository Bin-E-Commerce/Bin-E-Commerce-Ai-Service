"""Load LambdaRank artifact hợp lệ và báo chưa sẵn sàng nếu contract không khớp.

Registry sở hữu chi tiết LightGBM/metadata; khi artifact thiếu hoặc sai schema,
runtime trả trạng thái chưa sẵn sàng để API có thể fail closed.
"""

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from app.core.config import Settings
from app.modules.ranking.domain.models import (
    RANKING_FEATURE_MONOTONE_DIRECTIONS,
    RANKING_FEATURE_SCHEMA_VERSION,
)
from app.modules.ranking.infrastructure.training.ranking_evaluator import (
    RANKING_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


class LightGbmRankingModel:
    """Adapter load Booster LambdaRank đã có metadata tương ứng."""

    # Kiểm tra sidecar trước khi nạp model để classifier cũ hoặc sai schema không bị xem là ranker.
    def __init__(
        self,
        model_path: str,
        version: str,
        expected_features: int = 9,
    ) -> None:
        artifact_path = Path(model_path)
        metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("RANKING_MODEL_METADATA_UNAVAILABLE") from error
        # Model mới phải giữ đúng thứ tự cột và chiều ràng buộc; artifact cũ cần train lại trước khi serving.
        expected_constraints = [
            RANKING_FEATURE_MONOTONE_DIRECTIONS.get(feature_name, 0) for feature_name in RANKING_FEATURE_NAMES
        ]
        # Chỉ sidecar sinh bởi trainer LambdaRank mới được chấp nhận; weight cũ không chứng minh objective.
        if (
            not isinstance(metadata, dict)
            or metadata.get("objective") != "lambdarank"
            or metadata.get("featureCount") != expected_features
            or metadata.get("modelVersion") != version
            or metadata.get("featureSchemaVersion") != RANKING_FEATURE_SCHEMA_VERSION
            or metadata.get("featureNames") != list(RANKING_FEATURE_NAMES)
            or metadata.get("monotoneConstraints") != expected_constraints
        ):
            raise ValueError("RANKING_MODEL_METADATA_MISMATCH")

        try:
            import lightgbm as lgb
        except ImportError as error:
            raise RuntimeError("LIGHTGBM_NOT_INSTALLED") from error
        self._booster = lgb.Booster(model_file=str(artifact_path))
        # Sidecar không đủ nếu bị ghép nhầm; xác nhận objective thực được lưu trong artifact LightGBM.
        objective = self._booster.dump_model().get("objective", "")
        if not isinstance(objective, str) or not objective.strip().startswith("lambdarank"):
            raise ValueError("RANKING_MODEL_OBJECTIVE_MISMATCH")
        # Booster và metadata phải cùng số cột để cấm nạp artifact lệch với feature builder.
        if self._booster.num_feature() != expected_features:
            raise ValueError("RANKING_MODEL_FEATURE_COUNT_MISMATCH")
        self.model_version = version

    # Trả raw relevance score để thứ tự do LambdaRank học không bị co về xác suất giả trong [0, 1].
    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        import numpy as np

        values = self._booster.predict(np.asarray(rows, dtype=np.float32))
        scores = [float(value) for value in values]
        # Raw score vẫn phải hữu hạn để sort ổn định và tránh NaN làm comparator không xác định.
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("RANKING_MODEL_NON_FINITE_SCORE")
        return scores


# Chỉ đăng ký model có artifact và sidecar hợp lệ; không tự tạo scorer thay thế khi startup lỗi.
def build_ranking_model(settings: Settings) -> LightGbmRankingModel | None:
    """Nạp LambdaRank hoặc trả None để status/predict thể hiện model chưa sẵn sàng."""

    # Không path nghĩa là chưa deploy ranker; API predict sẽ phản hồi unavailable thay vì chấm giả.
    if not settings.ranking_model_path:
        logger.warning("Ranking model unavailable: artifact path is not configured")
        return None
    try:
        return LightGbmRankingModel(
            settings.ranking_model_path,
            settings.ranking_model_version,
            settings.ranking_expected_features,
        )
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("Ranking model unavailable: %s", error.__class__.__name__)
        return None
