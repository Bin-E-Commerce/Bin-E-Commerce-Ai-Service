"""Registry metrics tối giản cho AI Service.

File này chỉ lưu counter vận hành trong process và xuất Prometheus text format.
Không lưu secret, user identity, raw prompt, raw catalog text hoặc vector.
"""

from collections.abc import Mapping
from time import monotonic


# Registry chỉ chấp nhận các label có tập giá trị hữu hạn để tránh high-cardinality trong Prometheus.
class MetricsRegistry:
    """Thu thập counter an toàn cho HTTP API và worker trong một process."""

    # Khởi tạo registry riêng cho từng process để metric không bị chia sẻ ngầm giữa test và worker.
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._started_at = monotonic()

    # Tăng counter với label đã lọc; caller không thể vô tình đưa actor/product identifier vào metric.
    def increment(self, name: str, labels: Mapping[str, str] | None = None, amount: int = 1) -> None:
        allowed = {"status", "source", "surface", "variant", "error_code"}
        safe_labels = tuple(sorted((key, value) for key, value in (labels or {}).items() if key in allowed))
        key = (name, safe_labels)
        self._counters[key] = self._counters.get(key, 0) + amount

    # Xuất snapshot Prometheus mà không expose dữ liệu request hoặc provider response.
    def render(self) -> str:
        lines = [
            "# HELP ai_process_uptime_seconds Process uptime in seconds.",
            "# TYPE ai_process_uptime_seconds gauge",
            f"ai_process_uptime_seconds {int(monotonic() - self._started_at)}",
        ]
        for (name, labels), value in self._counters.items():
            label_text = ""
            if labels:
                label_text = "{" + ",".join(
                    f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels
                ) + "}"
            lines.append(f"{name}{label_text} {value}")
        return "\n".join(lines) + "\n"
