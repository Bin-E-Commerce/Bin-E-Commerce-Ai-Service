"""Mô hình dữ liệu domain cho batch ranking prediction."""

from dataclasses import dataclass


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
