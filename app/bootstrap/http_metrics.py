"""Chuẩn hóa label HTTP và ghi outcome vào registry metrics bounded."""

from app.core.metrics import MetricsRegistry

_KNOWN_HTTP_METHODS = frozenset({"CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"})


# Chuẩn hóa dữ liệu từ request thành label hữu hạn; route lạ không được đưa URL do client chọn vào metrics.
def normalize_http_metric_labels(
    method: str,
    route_template: str | None,
    status_code: int,
) -> dict[str, str]:
    normalized_method = method.upper()
    if normalized_method not in _KNOWN_HTTP_METHODS:
        normalized_method = "OTHER"

    # FastAPI route template chỉ có giá trị sau khi match; 404 dùng nhãn cố định để tránh cardinality theo URL.
    normalized_route = route_template[:160] if route_template else "unmatched"
    status_class = f"{status_code // 100}xx" if 100 <= status_code < 600 else "other"
    return {
        "method": normalized_method,
        "route": normalized_route,
        "status_class": status_class,
        "service": "ai-service",
    }


# Ghi cùng label set cho request, response và latency để dashboard không trộn dimension khác nhau.
def record_http_request_metrics(
    metrics: MetricsRegistry,
    method: str,
    route_template: str | None,
    status_code: int,
    duration_seconds: float,
) -> dict[str, str]:
    labels = normalize_http_metric_labels(method, route_template, status_code)
    metrics.increment("http_requests_total", labels)
    metrics.increment("http_responses_total", labels)
    metrics.observe_histogram("http_request_duration_seconds", duration_seconds, labels)
    return labels
