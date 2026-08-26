"""Adapter GPT-Image-2 cho output lifestyle, tach khoi use case de thay provider sau nay."""

import asyncio
import base64
import logging
from io import BytesIO
from typing import cast

import httpx
from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.core.errors import ConfigurationError, ProviderUnavailableError
from app.modules.image_optimization.domain.enums import LifestyleBackgroundPreset
from app.modules.image_optimization.domain.ports import GeneratedImage, LifestyleBackgroundRequest

logger = logging.getLogger(__name__)


class OpenAILifestyleImageProvider:
    """Goi image edit voi prompt an toan, khong log URL hay prompt raw."""

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise ConfigurationError()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            # Image generation thường lâu hơn text completion; timeout riêng tránh fail giả ở mốc 20 giây.
            timeout=settings.openai_image_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.openai_image_model
        self._quality = settings.openai_image_quality
        self._max_dimension = settings.ai_image_lifestyle_max_dimension
        self._jpeg_quality = settings.ai_image_lifestyle_jpeg_quality
        self._semaphore = asyncio.Semaphore(max(1, settings.ai_image_provider_max_concurrency))

    # Nhận bối cảnh đã kiểm soát thay vì dữ liệu request thô, giới hạn lời gọi trả phí bằng semaphore dùng chung worker.
    async def generate_lifestyle_background(
        self,
        source: bytes,
        file_name: str,
        request: LifestyleBackgroundRequest,
    ) -> GeneratedImage:
        """Gui bytes source da Media Service kiem soat toi GPT-Image-2 va tra output tam thoi."""

        try:
            # Gửi tuple có tên file và MIME rõ ràng; nếu chỉ truyền BytesIO, SDK mặc định octet-stream và OpenAI từ chối.
            # Giảm pixel và nén trước khi upload để rút ngắn latency nhưng vẫn đúng kích thước output 1024.
            image_file = self._build_openai_image_file(self._prepare_source(source))
            async with self._semaphore:
                result = await self._client.images.edit(
                    model=self._model,
                    image=image_file,
                    prompt=self._build_prompt(request),
                    n=1,
                    size="1024x1024",
                    quality=self._quality,
                    background="opaque",
                )
            items = result.data or []
            if not items:
                raise ProviderUnavailableError()
            item = items[0]
            if getattr(item, "b64_json", None):
                return GeneratedImage(
                    content=base64.b64decode(cast(str, item.b64_json)),
                    content_type="image/png",
                    file_name="lifestyle.png",
                )
            if getattr(item, "url", None):
                async with httpx.AsyncClient(timeout=15) as client:
                    generated = await client.get(cast(str, item.url))
                    generated.raise_for_status()
                return GeneratedImage(content=generated.content, content_type="image/png", file_name="lifestyle.png")
            raise ProviderUnavailableError()
        except (OpenAIError, httpx.HTTPError, TimeoutError, OSError) as error:
            # Chỉ ghi loại lỗi và status để vận hành chẩn đoán provider mà không làm lộ key, prompt, URL hay ảnh seller.
            logger.warning(
                "OpenAI image provider unavailable type=%s status=%s",
                type(error).__name__,
                getattr(error, "status_code", None),
            )
            raise ProviderUnavailableError() from error

    # Ghép preset và mô tả seller vào prompt cố định để model chỉ đổi nền, không bịa thêm sản phẩm hay nội dung bán hàng.
    @staticmethod
    def _build_prompt(request: LifestyleBackgroundRequest) -> str:
        """Tạo prompt tiếng Anh giới hạn phạm vi ảnh lifestyle theo chính sách marketplace."""

        preset_copy = {
            LifestyleBackgroundPreset.MINIMAL_STUDIO: "a minimal premium studio with soft neutral light",
            LifestyleBackgroundPreset.WARM_HOME: "a warm modern home setting with natural window light",
            LifestyleBackgroundPreset.NATURAL_OUTDOOR: "a refined outdoor setting with soft natural daylight",
            LifestyleBackgroundPreset.PREMIUM_DISPLAY: "a premium retail display with elegant controlled lighting",
        }
        selected_background = (
            preset_copy[request.preset] if request.preset is not None else "a clean premium commercial lifestyle setting"
        )
        custom_background = f" Seller requested background: {request.description}." if request.description else ""
        return (
            "Create one polished ecommerce lifestyle product photo. Preserve the exact product shape, color, logo, "
            "material, texture, proportions, and product count. Change only the background and lighting. "
            f"Use {selected_background}.{custom_background} "
            "Do not add people, hands, text, watermark, labels, claims, extra products, new logos, or product features."
        )

    # Chuẩn hóa ảnh source trong RAM để giảm payload đi qua mạng; ảnh gốc của seller vẫn do Media Service giữ nguyên.
    def _prepare_source(self, source: bytes) -> bytes:
        """Giảm kích thước và nén ảnh input trước khi gửi tới GPT-Image-2."""

        from PIL import Image, ImageOps

        with Image.open(BytesIO(source)) as image:
            prepared = ImageOps.exif_transpose(image)
            if prepared is None:
                raise OSError("Cannot prepare source image")
            prepared.thumbnail((self._max_dimension, self._max_dimension), Image.Resampling.LANCZOS)
            output = BytesIO()
            if "A" in prepared.getbands():
                prepared.convert("RGBA").save(output, format="PNG", optimize=True)
            else:
                prepared.convert("RGB").save(output, format="JPEG", quality=self._jpeg_quality, optimize=True)
            return output.getvalue()

    # Nhận diện MIME từ bytes thực tế để hỗ trợ ảnh JPEG/PNG/WebP bất kể tên file nội bộ từ Media Service.
    @staticmethod
    def _build_openai_image_file(source: bytes) -> tuple[str, bytes, str]:
        """Trả file tuple đúng contract multipart của OpenAI Images API."""

        from PIL import Image

        with Image.open(BytesIO(source)) as image:
            image_format = (image.format or "").upper()
        mime_by_format = {
            "JPEG": ("source.jpg", "image/jpeg"),
            "PNG": ("source.png", "image/png"),
            "WEBP": ("source.webp", "image/webp"),
        }
        try:
            file_name, mime_type = mime_by_format[image_format]
        except KeyError as error:
            raise OSError("Unsupported source image format") from error
        return file_name, source, mime_type
