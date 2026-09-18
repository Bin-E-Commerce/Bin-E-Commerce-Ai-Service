"""Tổng hợp baseline động và diễn biến từng ngày sau khi ảnh AI được áp dụng.

Use case này chỉ điều phối lifecycle job và gọi Recommendation Service qua port.
Nó không truy cập database analytics trực tiếp, không tạo dữ liệu giả và không thay đổi
trạng thái apply của ảnh sản phẩm.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from app.core.errors import AppError
from app.modules.image_optimization.application.ports import (
    ImageOptimizationJobRepository,
    ProductImpactComparison,
    ProductImpactDailyMetrics,
    ProductImpactMetrics,
    ProductImpactMetricsClient,
)
from app.modules.image_optimization.application.types.impact import (
    ImpactDailyPayload,
    ImpactMetricPayload,
    ProductImpactOverviewPayload,
    ProductImpactPayload,
    ProductImpactProductsPayload,
)
from app.modules.image_optimization.domain.enums import ImageOptimizationStatus
from app.modules.image_optimization.domain.models import ImageOptimizationJob


# Tính phần trăm an toàn; baseline bằng 0 không được biến thành phần trăm giả hoặc Infinity.
def _change_percent(before: float, after: float) -> float | None:
    """Trả phần trăm thay đổi giữa baseline và ngày quan sát gần nhất."""

    if before == 0:
        return None
    return round(((after - before) / before) * 100, 2)


# Metric trên card luôn so sánh trung bình mỗi ngày trước tối ưu với ngày sau mới nhất.
def _metric_payload(before: float, after: float) -> ImpactMetricPayload:
    """Chuẩn hóa số liệu thành shape frontend dùng chung cho summary và product row."""

    delta = round(after - before, 2)
    return {
        "before": round(before, 2),
        "after": round(after, 2),
        "delta": delta,
        "change_percent": _change_percent(before, after),
    }


# Map từng bucket ngày thành dữ liệu trình bày và giữ cùng baseline cho mọi ngày trong chuỗi.
def _daily_payload(
    daily: tuple[ProductImpactDailyMetrics, ...],
    baseline_views: float,
    baseline_sales: float,
) -> list[ImpactDailyPayload]:
    """Tạo chuỗi daily trend mà không làm frontend tự tính business metric."""

    return [
        {
            "date": item.bucket_date.isoformat(),
            "views": item.views,
            "sales": item.sales,
            "has_data": item.has_data,
            "views_change_percent": _change_percent(baseline_views, float(item.views)),
            "sales_change_percent": _change_percent(baseline_sales, float(item.sales)),
        }
        for item in daily
    ]


# Chỉ coi product có cả baseline và ngày hậu tối ưu là có metric để cộng vào summary.
def _is_metric_ready(item: ProductImpactPayload) -> bool:
    """Xác định product đã có baseline và ít nhất một ngày sau tối ưu để hiển thị số."""

    return item["views"] is not None and item["sales"] is not None


# Use case giữ ownership/lifecycle ở AI Service và chỉ đọc analytics qua typed client.
class GetImageOptimizationImpact:
    """Đọc baseline trước tối ưu và trend daily sau tối ưu cho seller.

    Số ngày baseline không cố định: Recommendation Service trả về toàn bộ các
    bucket lịch sử có dữ liệu trước ngày apply. Số bucket đó trở thành N, và
    use case chỉ so sánh N ngày đầu tiên sau khi ảnh được áp dụng.
    """

    # Nhận baseline/window từ composition root để production có thể đổi chính sách mà không sửa use case.
    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        metrics_client: ProductImpactMetricsClient | None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Nhận port persistence/analytics; số ngày baseline được lấy động từ dữ liệu thực tế."""

        self._repository = repository
        self._metrics_client = metrics_client
        self._clock = clock or (lambda: datetime.now(UTC))

    # Chỉ cộng product có baseline hợp lệ; product đang collecting vẫn có metric để seller demo ngay.
    async def overview(self, seller_owner_id: UUID) -> ProductImpactOverviewPayload:
        """Tổng hợp baseline trung bình và ngày hậu tối ưu mới nhất trên toàn seller."""

        jobs = await self._repository.find_latest_impact_jobs(seller_owner_id)
        products = await self._build_products(jobs)
        with_baseline = [item for item in products if _is_metric_ready(item)]
        ready = [item for item in products if item["status"] == "READY"]
        collecting = [item for item in products if item["status"] == "COLLECTING"]
        no_baseline = [item for item in products if item["status"] == "NO_BASELINE"]
        unavailable = [item for item in products if item["status"] == "UNAVAILABLE"]

        status = "READY" if ready else ("UNAVAILABLE" if unavailable and not with_baseline else "COLLECTING")
        return {
            "tracked_products": len(products),
            "ready_products": len(ready),
            "collecting_products": len(collecting),
            "no_baseline_products": len(no_baseline),
            "views": _metric_payload(
                sum(float(item["views"]["before"]) for item in with_baseline if item["views"]),
                sum(float(item["views"]["after"]) for item in with_baseline if item["views"]),
            ),
            "sales": _metric_payload(
                sum(float(item["sales"]["before"]) for item in with_baseline if item["sales"]),
                sum(float(item["sales"]["after"]) for item in with_baseline if item["sales"]),
            ),
            "status": status,
            "window_days": None,
            "baseline_days": None,
        }

    # Chỉ lấy product đã qua ownership check của job repository trước khi gọi Recommendation Service.
    async def products(self, seller_owner_id: UUID, product_ids: tuple[UUID, ...]) -> ProductImpactProductsPayload:
        """Trả trend theo đúng danh sách product seller đang nhìn thấy."""

        jobs = await self._repository.find_latest_impact_jobs(seller_owner_id)
        allowed = set(product_ids)
        selected_jobs = tuple(job for job in jobs if job.product_id in allowed)
        return {"items": await self._build_products(selected_jobs)}

    # Chuẩn bị baseline từ toàn bộ bucket trước apply và phần hậu tối ưu từ ngày apply tới hôm nay.
    # Ngày hiện tại được giữ trong range để seller thấy lượt xem demo ngay sau khi event Kafka được project.
    async def _build_products(
        self,
        jobs: tuple[ImageOptimizationJob, ...],
    ) -> list[ProductImpactPayload]:
        """Tạo trạng thái lifecycle trước, sau đó mới gọi analytics một lần theo batch."""

        now = self._clock().astimezone(UTC)
        pending: list[ProductImpactComparison] = []
        result_by_key: dict[str, ProductImpactPayload] = {}

        for job in jobs:
            key = f"{job.product_id}:{job.job_id}"
            if job.status is ImageOptimizationStatus.ROLLED_BACK:
                result_by_key[key] = {
                    "product_id": job.product_id,
                    "job_id": job.job_id,
                    "applied_at": job.completed_at,
                    "status": "ROLLED_BACK",
                    "elapsed_seconds": 0,
                    "days_collected": 0,
                    "window_days": 0,
                    "baseline_days": 0,
                    "views": None,
                    "sales": None,
                    "daily": [],
                }
                continue

            if job.status is not ImageOptimizationStatus.APPLIED or job.completed_at is None:
                continue

            applied_at = job.completed_at.astimezone(UTC)
            elapsed_seconds = max(0, int((now - applied_at).total_seconds()))
            result_by_key[key] = {
                "product_id": job.product_id,
                "job_id": job.job_id,
                "applied_at": job.completed_at,
                "status": "COLLECTING",
                "elapsed_seconds": elapsed_seconds,
                "days_collected": 0,
                "window_days": 0,
                "baseline_days": 0,
                "views": None,
                "sales": None,
                "daily": [],
            }

            if self._metrics_client is not None and applied_at <= now:
                pending.append(
                    ProductImpactComparison(
                        key=key,
                        product_id=job.product_id,
                        before_from=None,
                        before_to=applied_at,
                        after_from=applied_at,
                        after_to=now,
                    )
                )

        if not pending or self._metrics_client is None:
            if pending and self._metrics_client is None:
                for comparison in pending:
                    result_by_key[comparison.key]["status"] = "UNAVAILABLE"
            return list(result_by_key.values())

        try:
            metrics = await self._metrics_client.compare(tuple(pending))
        except AppError:
            # Upstream lỗi thì giữ lifecycle và không dựng số liệu thay thế để seller không hiểu sai báo cáo.
            for comparison in pending:
                result_by_key[comparison.key]["status"] = "UNAVAILABLE"
            return list(result_by_key.values())

        metrics_by_key: dict[str, ProductImpactMetrics] = {metric.key: metric for metric in metrics}
        for comparison in pending:
            item = result_by_key[comparison.key]
            metric = metrics_by_key.get(comparison.key)
            if metric is None or metric.product_id != comparison.product_id:
                item["status"] = "UNAVAILABLE"
                continue

            baseline_days = metric.before_days_with_data
            item["baseline_days"] = baseline_days
            item["window_days"] = baseline_days
            if metric.before_days_with_data == 0:
                item["days_collected"] = 0
                item["status"] = "NO_BASELINE"
                continue
            baseline_views = metric.before_average_views
            baseline_sales = metric.before_average_sales
            observed_daily = metric.daily[:baseline_days]
            item["daily"] = _daily_payload(observed_daily, baseline_views, baseline_sales)
            if not observed_daily:
                item["status"] = "COLLECTING"
                continue

            # Không đợi đủ N ngày mới hiển thị: chỉ cần đã có event sau mốc apply là seller thấy được dữ liệu thật.
            item["days_collected"] = metric.after_days_with_data
            latest = next((daily for daily in reversed(observed_daily) if daily.has_data), None)
            if latest is None:
                item["status"] = "COLLECTING"
                continue
            item["views"] = _metric_payload(baseline_views, float(latest.views))
            item["sales"] = _metric_payload(baseline_sales, float(latest.sales))
            item["status"] = "READY"

        return list(result_by_key.values())
