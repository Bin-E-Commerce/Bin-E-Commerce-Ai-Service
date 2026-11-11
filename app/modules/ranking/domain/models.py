"""Mô hình dữ liệu domain cho batch ranking prediction."""

from dataclasses import dataclass

# Schema version là contract code-owned để môi trường deploy không thể bật nhầm semantics feature cũ.
RANKING_FEATURE_SCHEMA_VERSION = "ranking-features-v2"

# Affinity không được làm giảm điểm khi các feature khác giữ nguyên; penalty âm không được làm tăng điểm.
RANKING_FEATURE_MONOTONE_DIRECTIONS = {
    "profileAffinity": 1,
    "negativePenalty": -1,
}


@dataclass(frozen=True)
class RankingPrediction:
    """Một score cho item, giữ item ID để Recommendation map lại đúng candidate."""

    item_id: str
    score: float


@dataclass(frozen=True)
class RankingPredictionBatch:
    """Kết quả batch cùng model version, không chứa user identity hay raw catalog."""

    request_id: str
    model_version: str
    predictions: tuple[RankingPrediction, ...]
