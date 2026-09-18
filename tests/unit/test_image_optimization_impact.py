"""Kiểm tra mapping trạng thái và cửa sổ so sánh của báo cáo impact ảnh AI."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.modules.image_optimization.application.ports import ProductImpactDailyMetrics, ProductImpactMetrics
from app.modules.image_optimization.application.use_cases import GetImageOptimizationImpact
from app.modules.image_optimization.domain.enums import (
    ImageOptimizationMode,
    ImageOptimizationStatus,
)
from app.modules.image_optimization.domain.models import ImageOptimizationJob

SELLER_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 9, 18, tzinfo=UTC)


def applied_job(
    *,
    product_id: UUID,
    completed_at: datetime,
    status: ImageOptimizationStatus = ImageOptimizationStatus.APPLIED,
) -> ImageOptimizationJob:
    """Tạo job tối thiểu đã có lifecycle để test không phụ thuộc persistence adapter."""

    job = ImageOptimizationJob.create(
        seller_owner_id=SELLER_ID,
        product_id=product_id,
        source_asset_ids=(uuid4(),),
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key=f"impact-{uuid4()}",
        expected_product_updated_at=NOW,
    )
    return replace(job, status=status, completed_at=completed_at, starts_impact_session=True)


class FakeRepository:
    """Repository fake chỉ trả lifecycle mới nhất mà use case cần đọc."""

    def __init__(self, jobs: tuple[ImageOptimizationJob, ...]) -> None:
        self.jobs = jobs

    async def find_latest_lifecycle_jobs(self, seller_owner_id: UUID) -> tuple[ImageOptimizationJob, ...]:
        """Không lọc thêm trong fake vì ownership đã là trách nhiệm của repository thật."""

        assert seller_owner_id == SELLER_ID
        return self.jobs

    # Impact query chỉ đọc job đã đánh dấu là phiên cover trong production repository.
    async def find_latest_impact_jobs(self, seller_owner_id: UUID) -> tuple[ImageOptimizationJob, ...]:
        """Trả cùng fixture để test use case tập trung vào phép tính metrics."""

        assert seller_owner_id == SELLER_ID
        return tuple(job for job in self.jobs if job.starts_impact_session)


class FakeMetricsClient:
    """Analytics fake trả aggregate cố định để kiểm tra phép tính trước/sau."""

    def __init__(self, metrics: tuple[ProductImpactMetrics, ...]) -> None:
        self.metrics = metrics
        self.calls = 0

    async def compare(self, comparisons):
        """Ghi nhận một batch call, bảo đảm use case không tạo N request."""

        self.calls += 1
        assert len(comparisons) == 1
        return self.metrics


@pytest.mark.asyncio
async def test_impact_overview_is_ready_after_full_window() -> None:
    """Đủ N ngày hậu tối ưu phải trả KPI trước/sau và phần trăm thay đổi an toàn."""

    product_id = uuid4()
    job = applied_job(product_id=product_id, completed_at=NOW - timedelta(days=30))
    metrics = FakeMetricsClient(
        (
            ProductImpactMetrics(
                key=f"{product_id}:{job.job_id}",
                product_id=product_id,
                before_views=100,
                before_sales=10,
                before_days_with_data=30,
                before_average_views=100,
                before_average_sales=10,
                after_views=125,
                after_sales=12,
                after_days_with_data=30,
                daily=(ProductImpactDailyMetrics(NOW.date(), 125, 12, True),),
            ),
        )
    )
    use_case = GetImageOptimizationImpact(FakeRepository((job,)), metrics, clock=lambda: NOW)

    result = await use_case.overview(SELLER_ID)

    assert result["status"] == "READY"
    assert result["ready_products"] == 1
    assert result["views"]["delta"] == 25
    assert result["views"]["change_percent"] == 25.0
    assert result["sales"]["delta"] == 2
    assert metrics.calls == 1


@pytest.mark.asyncio
async def test_impact_exposes_latest_event_without_waiting_for_a_full_day() -> None:
    """Có baseline và event hậu tối ưu thì seller thấy metric ngay theo timestamp thực tế."""

    product_id = uuid4()
    job = applied_job(product_id=product_id, completed_at=NOW)
    metrics = FakeMetricsClient(
        (
            ProductImpactMetrics(
                key=f"{product_id}:{job.job_id}",
                product_id=product_id,
                before_views=200,
                before_sales=20,
                before_days_with_data=2,
                before_average_views=100,
                before_average_sales=10,
                after_views=14,
                after_sales=2,
                after_days_with_data=1,
                daily=(ProductImpactDailyMetrics(NOW.date(), 14, 2, True),),
            ),
        )
    )
    use_case = GetImageOptimizationImpact(FakeRepository((job,)), metrics, clock=lambda: NOW)

    result = await use_case.products(SELLER_ID, (product_id,))

    item = result["items"][0]
    assert item["status"] == "READY"
    assert item["days_collected"] == 1
    assert item["window_days"] == 2
    assert item["baseline_days"] == 2
    assert item["views"] == {
        "before": 100,
        "after": 14,
        "delta": -86,
        "change_percent": -86.0,
    }
    assert len(item["daily"]) == 1


@pytest.mark.asyncio
async def test_impact_keeps_collecting_and_rollback_as_non_ready() -> None:
    """Sản phẩm chưa đủ N ngày hoặc đã rollback không được gộp vào KPI READY."""

    collecting = applied_job(product_id=uuid4(), completed_at=NOW - timedelta(days=5))
    rolled_back = applied_job(
        product_id=uuid4(),
        completed_at=NOW - timedelta(days=40),
        status=ImageOptimizationStatus.ROLLED_BACK,
    )
    use_case = GetImageOptimizationImpact(FakeRepository((collecting, rolled_back)), None, clock=lambda: NOW)

    result = await use_case.overview(SELLER_ID)

    assert result["status"] == "COLLECTING"
    assert result["collecting_products"] == 1
    assert result["ready_products"] == 0
    assert result["tracked_products"] == 2
