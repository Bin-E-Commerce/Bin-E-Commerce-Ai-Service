"""Typed response shapes cho baseline và trend daily của báo cáo tác động ảnh AI.

Các kiểu này thuộc application vì chúng mô tả dữ liệu use case trả cho presentation,
không phụ thuộc FastAPI, Pydantic hay database. Các field giữ nguyên tên snake_case để
schema HTTP có thể map alias ở đúng boundary thay vì rải chuỗi tự do trong use case.
"""

from datetime import datetime
from typing import TypedDict
from uuid import UUID


class ImpactMetricPayload(TypedDict):
    """Một metric baseline/ngày gần nhất đã được backend tính delta và phần trăm thay đổi."""

    before: float
    after: float
    delta: float
    change_percent: float | None


# Metric một ngày hậu tối ưu, giữ change percent đã tính ở application boundary.
class ImpactDailyPayload(TypedDict):
    """Một bucket sau tối ưu để frontend vẽ/hiển thị xu hướng theo từng ngày."""

    date: str
    views: int
    sales: int
    has_data: bool
    views_change_percent: float | None
    sales_change_percent: float | None


class ProductImpactPayload(TypedDict):
    """Trạng thái, baseline và trend daily của một product theo lifecycle job mới nhất."""

    product_id: UUID
    job_id: UUID | None
    applied_at: datetime | None
    status: str
    elapsed_seconds: int
    days_collected: int
    window_days: int
    baseline_days: int | None
    views: ImpactMetricPayload | None
    sales: ImpactMetricPayload | None
    daily: list[ImpactDailyPayload]


class ProductImpactOverviewPayload(TypedDict):
    """Summary impact đã gom theo seller.

    Mỗi product có thể có N baseline khác nhau, vì vậy summary không công bố
    một window chung; các metric vẫn được cộng từ baseline/ngày quan sát của
    từng product đã có dữ liệu.
    """

    tracked_products: int
    ready_products: int
    collecting_products: int
    no_baseline_products: int
    views: ImpactMetricPayload
    sales: ImpactMetricPayload
    status: str
    window_days: int | None
    baseline_days: int | None


class ProductImpactProductsPayload(TypedDict):
    """Response collection cho danh sách product impact trên dashboard."""

    items: list[ProductImpactPayload]
