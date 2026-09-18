"""Internal client đọc daily impact từ Recommendation Service.

Client này chỉ gửi product ID và ngày bucket đã được AI Service tạo; không gửi JWT,
URL ảnh hoặc dữ liệu người mua. Recommendation Service vẫn là source of truth cho
view/sales read model.
"""

from datetime import date
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.application.ports import (
    ProductImpactComparison,
    ProductImpactDailyMetrics,
    ProductImpactMetrics,
)
from app.modules.image_optimization.infrastructure.clients.base import InternalHttpClient


# Gọi endpoint analytics nội bộ với một request batch để không tạo N request theo từng sản phẩm.
class HttpProductImpactMetricsClient:
    """Map response daily analytics về dataclass application và che raw lỗi upstream."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Dùng shared AsyncClient của lifespan để tái sử dụng connection pool."""

        self._base_url = settings.recommendation_service_url.rstrip("/")
        self._token = settings.internal_service_token
        self._http = InternalHttpClient(settings.internal_service_token, client)

    # Gửi các cửa sổ so sánh đã được validate từ use case, không cho browser tự đặt range.
    async def compare(self, comparisons: tuple[ProductImpactComparison, ...]) -> tuple[ProductImpactMetrics, ...]:
        """Trả empty tuple khi không có comparison và map response không tin cậy thành 503 nội bộ."""

        if not comparisons:
            return ()
        payload = {
            "comparisons": [
                {
                    "key": item.key,
                    "productId": str(item.product_id),
                    "beforeFrom": item.before_from.isoformat() if item.before_from else None,
                    "beforeTo": item.before_to.isoformat(),
                    "afterFrom": item.after_from.isoformat(),
                    "afterTo": item.after_to.isoformat(),
                }
                for item in comparisons
            ]
        }
        headers = {
            "x-internal-service-token": self._token.get_secret_value() if self._token else "",
        }
        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/api/v1/internal/analytics/product-impact",
                timeout=10,
                headers=headers,
                json=payload,
            )
            self._http.ensure_success(response)
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("items"), list):
                raise ProviderUnavailableError()
            return tuple(self._to_metrics(item) for item in body["items"])
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError) as error:
            raise ProviderUnavailableError() from error

    # Parse từng item bằng allow-list field để upstream không thể đẩy dữ liệu ngoài contract vào application.
    @staticmethod
    def _to_metrics(value: object) -> ProductImpactMetrics:
        """Chuyển JSON primitive thành metrics typed và fail closed khi thiếu field."""

        if not isinstance(value, dict):
            raise ProviderUnavailableError()
        try:
            before = value["before"]
            after = value["after"]
            return ProductImpactMetrics(
                key=str(value["key"]),
                product_id=UUID(str(value["productId"])),
                before_views=int(before["views"]),
                before_sales=int(before["sales"]),
                before_days_with_data=int(before["daysWithData"]),
                before_average_views=float(before["averageViews"]),
                before_average_sales=float(before["averageSales"]),
                after_views=int(after["views"]),
                after_sales=int(after["sales"]),
                after_days_with_data=int(after["daysWithData"]),
                daily=tuple(HttpProductImpactMetricsClient._to_daily_metric(item) for item in value["daily"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderUnavailableError() from error

    # Parse daily trend với allow-list để upstream không thể chèn field ngoài contract vào application.
    @staticmethod
    def _to_daily_metric(value: object) -> ProductImpactDailyMetrics:
        """Chuyển một bucket ngày thành dataclass và fail closed nếu thiếu dữ liệu."""

        if not isinstance(value, dict):
            raise ProviderUnavailableError()
        return ProductImpactDailyMetrics(
            bucket_date=date.fromisoformat(str(value["date"])),
            views=int(value["views"]),
            sales=int(value["sales"]),
            has_data=bool(value["hasData"]),
        )
