"""Adapter tạo nền trắng bằng rembg và Pillow.

Provider không gọi LLM, không ghi file và không được phép trả ảnh gốc dưới tên
ảnh đã tối ưu. Thiếu model hoặc tách nền thất bại luôn được báo lỗi rõ ràng.
"""

import asyncio
from collections.abc import Callable
from io import BytesIO
from threading import Lock
from typing import Any

from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.application.ports import GeneratedImage, ProviderExecutionMetadata


# Xử lý ảnh CPU-bound trong thread và dùng chung một rembg session đã warm.
class WhiteBackgroundProvider:
    """Tách chủ thể thật sự trước khi ghép canvas trắng và xuất WebP."""

    # Chuẩn hóa giới hạn xử lý để tránh ảnh quá lớn làm cạn RAM worker.
    def __init__(self, max_dimension: int = 2048, webp_quality: int = 88) -> None:
        """Lưu cấu hình an toàn; model chỉ được khởi tạo khi warm-up hoặc xử lý."""

        self._max_dimension = max(512, max_dimension)
        self._webp_quality = min(95, max(60, webp_quality))
        self._rembg_session: Any | None = None
        self._session_lock = Lock()

    # Tải model trước khi worker nhận message để lỗi cấu hình xuất hiện ngay khi khởi động.
    async def warm_up(self) -> None:
        """Fail-fast nếu rembg hoặc model không sẵn sàng, thay vì tạo false-success."""

        try:
            await asyncio.to_thread(self._get_rembg_components)
        except (ImportError, OSError, RuntimeError) as error:
            raise ProviderUnavailableError() from error

    # Chuyển pipeline đồng bộ sang thread để không khóa event loop Kafka và HTTP.
    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        """Trả ảnh WebP thật sự đã tách nền hoặc raise lỗi provider ổn định."""

        try:
            return await asyncio.to_thread(self._generate_sync, source, file_name)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
            raise ProviderUnavailableError() from error

    # Chuẩn hóa orientation, color mode và kích thước trước khi chạy segmentation.
    def _prepare_source(self, source: bytes) -> bytes:
        """Chặn dữ liệu không phải ảnh và giảm số pixel phải phân tích."""

        from PIL import Image, ImageOps, UnidentifiedImageError

        if not source:
            raise ValueError("Source image is empty")
        try:
            with Image.open(BytesIO(source)) as image:
                transposed = ImageOps.exif_transpose(image)
                if transposed is None:
                    raise OSError("Cannot normalize source image")
                prepared = transposed.convert("RGBA")
                if max(prepared.size) > self._max_dimension:
                    prepared.thumbnail((self._max_dimension, self._max_dimension), Image.Resampling.LANCZOS)
                output = BytesIO()
                prepared.save(output, format="PNG", optimize=True)
                return output.getvalue()
        except UnidentifiedImageError as error:
            raise ValueError("Unsupported source image") from error

    # Khởi tạo đúng một rembg session để hai job đầu tiên không tải model trùng.
    def _get_rembg_components(self) -> tuple[Callable[..., bytes], Any]:
        """Không cung cấp fallback vì không có segmentation thì chưa phải tối ưu nền trắng."""

        from rembg import new_session, remove

        if self._rembg_session is None:
            with self._session_lock:
                if self._rembg_session is None:
                    self._rembg_session = new_session("u2net")
        return remove, self._rembg_session

    # Chạy segmentation, xác minh alpha mask rồi mới ghép nền trắng.
    def _generate_sync(self, source: bytes, file_name: str) -> GeneratedImage:
        """Bảo đảm output khác pipeline ảnh gốc và có MIME/file extension nhất quán."""

        from PIL import Image

        prepared = self._prepare_source(source)
        remove, session = self._get_rembg_components()
        processed = remove(prepared, session=session)
        if not processed:
            raise RuntimeError("Background segmentation returned no image")

        with Image.open(BytesIO(processed)) as foreground:
            rgba_foreground = foreground.convert("RGBA")
            alpha = rgba_foreground.getchannel("A")
            minimum_alpha, maximum_alpha = alpha.getextrema()
            # Alpha toàn 255 nghĩa là model không tách được nền; báo lỗi để seller không nhận ảnh giả tối ưu.
            if minimum_alpha == maximum_alpha == 255:
                raise RuntimeError("Background segmentation did not produce an alpha mask")
            canvas = Image.new("RGBA", rgba_foreground.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba_foreground)
            output = BytesIO()
            canvas.convert("RGB").save(output, format="WEBP", quality=self._webp_quality, method=4)

        stem = file_name.rsplit(".", 1)[0]
        return GeneratedImage(
            content=output.getvalue(),
            content_type="image/webp",
            file_name=f"{stem}-white.webp",
            metadata=ProviderExecutionMetadata(
                provider="white-background-local",
                model="rembg-u2net",
                prompt_version=None,
            ),
        )
