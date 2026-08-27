"""Đăng ký provider theo capability để processor không phụ thuộc vendor cụ thể."""

from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.application.ports import (
    GeneratedImage,
    LifestyleBackgroundProviderPort,
    LifestyleBackgroundRequest,
    WhiteBackgroundProviderPort,
)
from app.modules.image_optimization.domain.enums import ImageOptimizationMode


# Registry là điểm duy nhất ánh xạ mode nghiệp vụ sang capability provider đã được bootstrap cấu hình.
class ImageOptimizationProviderRegistry:
    """Điều phối provider theo capability mà không hard-code vendor hoặc model trong use case."""

    # Nhận từng capability riêng để provider không bị ép cài method nó không hỗ trợ.
    def __init__(
        self,
        *,
        white_background: WhiteBackgroundProviderPort,
        lifestyle_background: LifestyleBackgroundProviderPort | None,
    ) -> None:
        """Lưu adapter đã được composition root lựa chọn từ typed settings."""

        self._white_background = white_background
        self._lifestyle_background = lifestyle_background

    # Chọn capability theo mode và giữ mọi vendor branching bên ngoài processor.
    async def generate(
        self,
        *,
        mode: ImageOptimizationMode,
        source: bytes,
        file_name: str,
        lifestyle_request: LifestyleBackgroundRequest,
    ) -> GeneratedImage:
        """Sinh một output hoặc raise lỗi ổn định khi capability chưa được cấu hình."""

        if mode is ImageOptimizationMode.WHITE_BACKGROUND:
            return await self._white_background.generate_white_background(source, file_name)
        if mode is ImageOptimizationMode.LIFESTYLE_BACKGROUND and self._lifestyle_background is not None:
            return await self._lifestyle_background.generate_lifestyle_background(source, file_name, lifestyle_request)
        raise ProviderUnavailableError()

    # Cho processor biết pipeline nào là local để chỉ pipeline miễn phí mới được retry.
    @staticmethod
    def is_local(mode: ImageOptimizationMode) -> bool:
        """Trả True duy nhất cho nền trắng rembg/Pillow không tính phí theo request."""

        return mode is ImageOptimizationMode.WHITE_BACKGROUND
