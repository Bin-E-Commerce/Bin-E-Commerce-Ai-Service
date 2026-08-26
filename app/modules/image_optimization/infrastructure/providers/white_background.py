"""Provider tạo nền trắng local, tối ưu CPU và không gửi ảnh sang LLM.

Provider giữ một rembg session đã warm trong worker, chuyển phần xử lý đồng bộ sang thread
để không chặn event loop Kafka. Provider không lưu ảnh, không ghi database và không thay đổi asset gốc.
"""

import asyncio
from collections.abc import Callable
from io import BytesIO
from threading import Lock
from typing import Any

from app.modules.image_optimization.domain.ports import GeneratedImage


# Adapter local ưu tiên tốc độ cho output nền trắng, không gọi provider trả phí.
class WhiteBackgroundProvider:
    """Tách nền bằng rembg/Pillow với model warm và giới hạn kích thước đầu vào."""

    # Khởi tạo cấu hình xử lý; session rembg được tạo lười để service vẫn chạy khi model chưa cài.
    def __init__(self, max_dimension: int = 2048, webp_quality: int = 88) -> None:
        self._max_dimension = max(512, max_dimension)
        self._webp_quality = min(95, max(60, webp_quality))
        self._rembg_session: Any | None = None
        self._session_lock = Lock()

    # Warm model trước message đầu tiên để độ trễ request đầu không bao gồm thời gian tải model rembg.
    async def warm_up(self) -> None:
        try:
            await asyncio.to_thread(self._get_rembg_components)
        except (ImportError, OSError, RuntimeError):
            # Worker vẫn khởi động được khi rembg/model chưa sẵn sàng; lần xử lý sau sẽ dùng fallback Pillow.
            return

    # Chuyển xử lý ảnh CPU-bound ra thread để relay Kafka và các coroutine HTTP không bị khóa event loop.
    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        return await asyncio.to_thread(self._generate_sync, source, file_name)

    # Chuẩn hóa ảnh trước khi tách nền để giảm số pixel rembg phải phân tích mà vẫn giữ ảnh gốc nguyên vẹn.
    def _prepare_source(self, source: bytes) -> bytes:
        from PIL import Image

        with Image.open(BytesIO(source)) as image:
            prepared = image.convert("RGBA")
            if max(prepared.size) > self._max_dimension:
                prepared.thumbnail((self._max_dimension, self._max_dimension), Image.Resampling.LANCZOS)
            output = BytesIO()
            prepared.save(output, format="PNG", optimize=True)
            return output.getvalue()

    # Trả hàm remove và session dùng chung; lock bảo đảm hai job đầu tiên không khởi tạo model trùng nhau.
    def _get_rembg_components(self) -> tuple[Callable[..., bytes] | None, Any | None]:
        try:
            from rembg import new_session, remove
        except ImportError:
            return None, None

        if self._rembg_session is None:
            with self._session_lock:
                if self._rembg_session is None:
                    self._rembg_session = new_session("u2net")
        return remove, self._rembg_session

    # Thực hiện toàn bộ pipeline đồng bộ trong thread, chỉ trả binary tạm thời cho Media Service upload.
    def _generate_sync(self, source: bytes, file_name: str) -> GeneratedImage:
        try:
            from PIL import Image

            prepared = self._prepare_source(source)
            remove, session = self._get_rembg_components()
            processed = remove(prepared, session=session) if remove is not None and session is not None else prepared
            with Image.open(BytesIO(processed)) as foreground:
                canvas = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
                canvas.alpha_composite(foreground.convert("RGBA"))
                output = BytesIO()
                canvas.convert("RGB").save(output, format="WEBP", quality=self._webp_quality, method=4)
            stem = file_name.rsplit(".", 1)[0]
            return GeneratedImage(content=output.getvalue(), content_type="image/webp", file_name=f"{stem}-white.webp")
        except (ImportError, OSError, ValueError, RuntimeError):
            # Không làm mất ảnh gốc nếu model/ảnh lỗi; worker vẫn đánh dấu output để seller xem xét an toàn.
            return GeneratedImage(content=source, content_type="application/octet-stream", file_name=f"{file_name}-white")
