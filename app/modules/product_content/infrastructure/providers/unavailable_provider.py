"""Provider placeholder cho memory mode khi local chưa cấu hình OpenAI key."""

from collections.abc import Sequence

from app.core.errors import ConfigurationError
from app.modules.product_content.domain.models import GeneratedName, ProductContext


class UnavailableProductContentProvider:
    """Giữ composition root khởi động được nhưng không giả mạo kết quả AI."""

    # Memory mode phục vụ health/ranking local; content generation phải báo cấu hình thiếu thay vì trả dữ liệu giả.
    async def generate_name_suggestions(self, context: ProductContext) -> Sequence[GeneratedName]:
        """Từ chối rõ ràng khi product-content provider chưa được cấu hình."""

        raise ConfigurationError()

    # Memory mode không được âm thầm sinh description giả vì seller có thể lưu nhầm vào product thật.
    async def generate_description(self, context: ProductContext) -> str:
        """Từ chối rõ ràng khi product-content provider chưa được cấu hình."""

        raise ConfigurationError()
