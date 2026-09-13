"""Protocol để application không phụ thuộc LightGBM hoặc model artifact cụ thể."""

from collections.abc import Sequence
from typing import Protocol


class RankingModel(Protocol):
    """Capability dự đoán score cho một batch feature vector đã chuẩn hóa."""

    @property
    def model_version(self) -> str:
        """Trả model version đi kèm response để Recommendation audit/cache."""

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Dự đoán score theo đúng thứ tự rows."""
