"""Registry LightGBM và fallback model; không để provider detail lọt vào application layer."""

import logging
from collections.abc import Sequence

from app.core.config import Settings

logger = logging.getLogger(__name__)


class FallbackRankingModel:
    """Scorer deterministic dùng khi chưa có artifact hoặc LightGBM chưa sẵn sàng."""

    model_version = "ranking-fallback-v1"

    # Fallback giữ score trong [0, 1] để Hybrid blend không đổi semantics khi AI unavailable.
    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Tính trung bình các feature hợp lệ, không gọi network/provider."""

        return [
            max(
                0.0,
                min(
                    1.0,
                    sum(row[:-1] if len(row) > 1 else row) / max(1, len(row[:-1] if len(row) > 1 else row))
                    - (row[-1] if len(row) > 1 else 0.0),
                ),
            )
            for row in rows
        ]


class LightGbmRankingModel:
    """Adapter load Booster từ artifact local, chỉ import LightGBM ở infrastructure."""

    def __init__(self, model_path: str, version: str, expected_features: int = 9) -> None:
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise RuntimeError("LIGHTGBM_NOT_INSTALLED") from error
        self._booster = lgb.Booster(model_file=model_path)
        if self._booster.num_feature() != expected_features:
            raise ValueError("RANKING_MODEL_FEATURE_COUNT_MISMATCH")
        self.model_version = version

    # Gọi Booster theo batch và clamp output để response không phát score bất thường do artifact lỗi.
    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        values = self._booster.predict(rows)
        return [max(0.0, min(1.0, float(value))) for value in values]


def build_ranking_model(settings: Settings) -> LightGbmRankingModel | FallbackRankingModel:
    """Load ML artifact nếu cấu hình đủ; fallback deterministic giúp API vẫn phục vụ an toàn."""

    if not settings.ranking_model_path:
        return FallbackRankingModel()
    try:
        return LightGbmRankingModel(
            settings.ranking_model_path,
            settings.ranking_model_version,
            settings.ranking_expected_features,
        )
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("Ranking model unavailable; using fallback: %s", error.__class__.__name__)
        return FallbackRankingModel()
