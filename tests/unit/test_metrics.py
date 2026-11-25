"""Kiểm thử registry Prometheus; xác nhận scrape nhất quán khi có nhiều thread ghi metric."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

from app.bootstrap.http_metrics import record_http_request_metrics
from app.core.metrics import MetricsRegistry


# Mỗi metric family chỉ khai báo TYPE một lần dù có nhiều bộ label khác nhau.
def test_metrics_registry_emits_one_type_per_metric_family() -> None:
    registry = MetricsRegistry()
    registry.increment("image_embedding_total", {"variant": "success"})
    registry.increment("image_embedding_total", {"variant": "failure"})
    registry.observe_histogram("image_embedding_duration_seconds", 0.1, {"service": "ai"})
    registry.observe_histogram("image_embedding_duration_seconds", 0.2, {"service": "product"})

    rendered = registry.render()

    assert rendered.count("# TYPE image_embedding_total counter") == 1
    assert rendered.count("# TYPE image_embedding_duration_seconds histogram") == 1
    assert 'image_embedding_total{variant="failure"} 1' in rendered
    assert 'image_embedding_total{variant="success"} 1' in rendered
    assert 'image_embedding_duration_seconds_count{service="ai"} 1' in rendered
    assert 'image_embedding_duration_seconds_count{service="product"} 1' in rendered


# Label escape giữ exposition trên một dòng và không làm hỏng backslash hoặc dấu quote gốc.
def test_metrics_registry_escapes_newlines_quotes_and_backslashes() -> None:
    registry = MetricsRegistry()
    registry.increment("escaped_labels_total", {"source": 'quote"\nline\\tail'})

    rendered = registry.render()

    assert 'escaped_labels_total{source="quote\\"\\nline\\\\tail"} 1' in rendered


# 404 và method extension phải dùng label hữu hạn, không lấy nguyên URL hoặc token method từ client.
def test_http_metrics_normalize_unmatched_route_and_unknown_method() -> None:
    registry = MetricsRegistry()

    labels = record_http_request_metrics(
        registry,
        "CUSTOM-VERB-123",
        None,
        404,
        0.01,
    )
    rendered = registry.render()

    assert labels == {
        "method": "OTHER",
        "route": "unmatched",
        "status_class": "4xx",
        "service": "ai-service",
    }
    assert 'http_requests_total{method="OTHER",route="unmatched",service="ai-service",status_class="4xx"} 1' in rendered


# Counter và histogram phải giữ đủ cập nhật đồng thời từ worker thread mà không mất sample.
def test_metrics_registry_preserves_concurrent_counter_and_histogram_updates() -> None:
    registry = MetricsRegistry()
    worker_count = 8
    updates_per_worker = 100
    start_barrier = Barrier(worker_count + 1)
    writes_finished = Event()

    # Đồng bộ điểm bắt đầu; mỗi worker chỉ tạo một route series cố định trong khi scrape.
    def record_metrics(worker_index: int) -> None:
        start_barrier.wait()
        for _update_index in range(updates_per_worker):
            registry.increment("parallel_requests_total", {"service": "ai"})
            registry.observe_histogram(
                "parallel_latency_seconds",
                0.01,
                {"route": f"/test/{worker_index}"},
            )

    # Scrape trong lúc các worker thêm label set để render không duyệt dictionary đang thay đổi.
    def scrape_while_writing() -> None:
        start_barrier.wait()
        while not writes_finished.is_set():
            registry.render()

    with ThreadPoolExecutor(max_workers=worker_count + 1) as executor:
        scrapes = executor.submit(scrape_while_writing)
        writers = [executor.submit(record_metrics, worker_index) for worker_index in range(worker_count)]
        for writer in writers:
            writer.result()
        writes_finished.set()
        scrapes.result()

    rendered = registry.render()
    expected_updates = worker_count * updates_per_worker
    histogram_counts = [
        int(line.rsplit(" ", 1)[1])
        for line in rendered.splitlines()
        if line.startswith('parallel_latency_seconds_count{route="/test/')
    ]

    assert f'parallel_requests_total{{service="ai"}} {expected_updates}' in rendered
    assert rendered.count("# TYPE parallel_latency_seconds histogram") == 1
    assert len(histogram_counts) == worker_count
    assert sum(histogram_counts) == expected_updates
