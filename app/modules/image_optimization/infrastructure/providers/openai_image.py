"""Adapter GPT-Image-2 cho output lifestyle, tach khoi use case de thay provider sau nay."""

import asyncio
import base64
import logging
from io import BytesIO
from typing import cast
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.core.errors import ConfigurationError, ProviderUnavailableError
from app.modules.image_optimization.application.ports import (
    GeneratedImage,
    LifestyleBackgroundRequest,
    ProviderExecutionMetadata,
)
from app.modules.image_optimization.application.prompts import LIFESTYLE_PROMPT_VERSION, build_lifestyle_prompt

logger = logging.getLogger(__name__)


class OpenAILifestyleImageProvider:
    """Goi image edit voi prompt an toan, khong log URL hay prompt raw."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
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
        self._http_client = http_client
        self._allowed_output_hosts = frozenset(
            value.strip().lower() for value in settings.openai_image_output_hosts.split(",") if value.strip()
        )
        self._max_output_bytes = settings.ai_image_generated_max_bytes

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
                    prompt=build_lifestyle_prompt(request),
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
                    metadata=self._execution_metadata(),
                )
            if getattr(item, "url", None):
                generated_content = await self._download_generated_image(cast(str, item.url))
                return GeneratedImage(
                    content=generated_content,
                    content_type="image/png",
                    file_name="lifestyle.png",
                    metadata=self._execution_metadata(),
                )
            raise ProviderUnavailableError()
        except (OpenAIError, httpx.HTTPError, TimeoutError, OSError) as error:
            # Chỉ ghi loại lỗi và status để vận hành chẩn đoán provider mà không làm lộ key, prompt, URL hay ảnh seller.
            logger.warning(
                "OpenAI image provider unavailable type=%s status=%s",
                type(error).__name__,
                getattr(error, "status_code", None),
            )
            raise ProviderUnavailableError() from error

    # Trả metadata từ cấu hình adapter để application không biết tên model OpenAI cụ thể.
    def _execution_metadata(self) -> ProviderExecutionMetadata:
        """Gắn vendor/model/prompt version chính xác vào output phục vụ audit."""

        return ProviderExecutionMetadata(
            provider="openai",
            model=self._model,
            prompt_version=LIFESTYLE_PROMPT_VERSION,
        )

    # Chỉ tải URL do provider trả từ allow-list, giới hạn byte và xác minh magic bytes bằng Pillow.
    async def _download_generated_image(self, url: str) -> bytes:
        """Ngăn SSRF, response quá lớn hoặc nội dung giả ảnh đi vào Media Service."""

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in self._allowed_output_hosts:
            raise ProviderUnavailableError()
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=15, follow_redirects=False)
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"image/png", "image/jpeg", "image/webp"}:
                    raise ProviderUnavailableError()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_output_bytes:
                        raise ProviderUnavailableError()
                    chunks.append(chunk)
            content = b"".join(chunks)
            from PIL import Image

            with Image.open(BytesIO(content)) as image:
                image.verify()
            return content
        except (httpx.HTTPError, OSError) as error:
            raise ProviderUnavailableError() from error
        finally:
            if owns_client:
                await client.aclose()

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
