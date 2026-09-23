"""Registry metrics nội bộ cho AI Service.

File này chỉ lưu metric vận hành có label bounded và xuất Prometheus text format.
File không lưu secret, user identity, prompt, raw catalog text hoặc vector.
"""

import os
from collections.abc import Mapping
from time import monotonic
from typing import TypedDict


class HistogramSample(TypedDict):
    buckets: list[float]
    counts: list[int]
    sum: float
    count: int


# Escape label value và giữ thứ tự ổn định trước khi xuất Prometheus text format.
def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(
        f'{key}="{value.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34))}"' for key, value in labels
    )
    return "{" + rendered + "}"


# Đọc RSS từ process hiện tại để dashboard phát hiện worker phình bộ nhớ mà không gọi hệ thống ngoài.
def _resident_memory_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        sysconf = getattr(os, "sysconf", None)
        page_size = int(sysconf("SC_PAGE_SIZE")) if callable(sysconf) else 4096
        return resident_pages * page_size
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return 0


# Registry chỉ nhận label bounded để tránh mỗi user/product tạo một series riêng trong Prometheus.
class MetricsRegistry:
    """Thu thập counter, gauge và histogram nhẹ cho HTTP API và worker."""

    # Khởi tạo registry riêng cho từng process để test và worker không chia sẻ state ngầm.
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], HistogramSample] = {}
        self._started_at = monotonic()

    # Tăng counter với label đã lọc; caller không thể vô tình đưa actor/product identifier vào metric.
    def increment(self, name: str, labels: Mapping[str, str] | None = None, amount: int = 1) -> None:
        allowed = {
            "service",
            "method",
            "route",
            "status",
            "status_class",
            "source",
            "surface",
            "variant",
            "error_code",
            "topic",
            "consumer",
            "dependency",
        }
        safe_labels = tuple(sorted((key, value) for key, value in (labels or {}).items() if key in allowed))
        key = (name, safe_labels)
        self._counters[key] = self._counters.get(key, 0) + amount

    # Ghi giá trị hiện tại cho dependency hoặc một trạng thái vận hành bounded.
    def set_gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        allowed = {"service", "dependency", "consumer", "topic"}
        safe_labels = tuple(sorted((key, value) for key, value in (labels or {}).items() if key in allowed))
        self._gauges[(name, safe_labels)] = value

    # Cộng hoặc trừ gauge trong cùng event loop để phản ánh request đồng thời chính xác.
    def add_gauge(self, name: str, amount: float, labels: Mapping[str, str] | None = None) -> None:
        allowed = {"service", "dependency", "consumer", "topic"}
        safe_labels = tuple(sorted((key, value) for key, value in (labels or {}).items() if key in allowed))
        key = (name, safe_labels)
        self._gauges[key] = max(0.0, self._gauges.get(key, 0.0) + amount)

    # Lưu histogram bounded để Prometheus tính p95 mà không giữ raw request hoặc payload AI.
    def observe_histogram(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        allowed = {"service", "method", "route", "status_class"}
        safe_labels = tuple(sorted((key, value) for key, value in (labels or {}).items() if key in allowed))
        key = (name, safe_labels)
        buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        sample = self._histograms.setdefault(
            key,
            {"buckets": buckets, "counts": [0] * len(buckets), "sum": 0.0, "count": 0},
        )
        safe_value = value if value >= 0 else 0.0
        sample["sum"] = float(sample["sum"]) + safe_value
        sample["count"] = int(sample["count"]) + 1
        sample["counts"] = [
            count + (1 if safe_value <= bucket else 0) for bucket, count in zip(sample["buckets"], sample["counts"], strict=True)
        ]

    # Xuất snapshot Prometheus mà không expose dữ liệu request, provider response hoặc identity.
    def render(self) -> str:
        lines = [
            "# HELP ai_process_uptime_seconds Process uptime in seconds.",
            "# TYPE ai_process_uptime_seconds gauge",
            f"ai_process_uptime_seconds {int(monotonic() - self._started_at)}",
            "# HELP ai_process_resident_memory_bytes Resident process memory in bytes.",
            "# TYPE ai_process_resident_memory_bytes gauge",
            f"ai_process_resident_memory_bytes {_resident_memory_bytes()}",
        ]
        for (name, labels), counter_value in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{_render_labels(labels)} {counter_value}")
        for (name, labels), gauge_value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{_render_labels(labels)} {gauge_value}")
        for (name, labels), sample in self._histograms.items():
            lines.append(f"# TYPE {name} histogram")
            for bucket, count in zip(sample["buckets"], sample["counts"], strict=True):
                lines.append(f"{name}_bucket{_render_labels((*labels, ('le', str(bucket))))} {count}")
            lines.append(f"{name}_bucket{_render_labels((*labels, ('le', '+Inf')))} {sample['count']}")
            lines.append(f"{name}_sum{_render_labels(labels)} {sample['sum']}")
            lines.append(f"{name}_count{_render_labels(labels)} {sample['count']}")
        return "\n".join(lines) + "\n"
