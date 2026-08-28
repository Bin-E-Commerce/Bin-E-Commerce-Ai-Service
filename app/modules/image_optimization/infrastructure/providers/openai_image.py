"""Adapter OpenAI cho ảnh lifestyle.

File này cô lập SDK OpenAI, chuẩn hóa ảnh đầu vào và kiểm tra ảnh đầu ra.
Nó không quyết định nghiệp vụ, không lưu prompt/ảnh và không được làm lộ secret
hay chi tiết lỗi nội bộ ra API.
"""

import asyncio
import base64
import logging
import time
from io import BytesIO
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from openai import APITimeoutError, AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.core.errors import (
    ConfigurationError,
    ProviderConfigurationError,
    ProviderRateLimitedError,
    ProviderRequestRejectedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.modules.image_optimization.application.ports import (
    GeneratedImage,
    LifestyleBackgroundRequest,
    ProviderExecutionMetadata,
)
from app.modules.image_optimization.application.prompts import LIFESTYLE_PROMPT_VERSION, build_lifestyle_prompt
from app.modules.image_optimization.domain.enums import ImageGenerationProfile

logger = logging.getLogger(__name__)


class OpenAILifestyleImageProvider:
    """Gọi image edit với prompt an toàn, không log URL hay prompt raw."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise ConfigurationError()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            # Image generation thường lâu hơn text completion; timeout riêng tránh fail giả ở mốc 20 giây.
            timeout=settings.openai_image_timeout_seconds,
            max_retries=0,
            # Dùng connection pool chung của worker để tránh tạo TCP/TLS connection cho từng output.
            http_client=http_client,
        )
        self._model = settings.openai_image_model
        self._profiles: dict[ImageGenerationProfile, dict[str, Any]] = {
            ImageGenerationProfile.PREVIEW: {
                "quality": settings.ai_image_preview_quality,
                "size": settings.ai_image_preview_size,
                "format": settings.ai_image_preview_format,
                "compression": settings.ai_image_preview_compression,
                "max_dimension": settings.ai_image_preview_max_dimension,
                "input_fidelity": settings.ai_image_preview_input_fidelity,
                "timeout": settings.ai_image_preview_timeout_seconds,
            },
            ImageGenerationProfile.FINAL: {
                "quality": settings.ai_image_final_quality,
                "size": settings.ai_image_final_size,
                "format": settings.ai_image_final_format,
                "compression": settings.ai_image_final_compression,
                "max_dimension": settings.ai_image_final_max_dimension,
                "input_fidelity": settings.ai_image_final_input_fidelity,
                "timeout": settings.ai_image_final_timeout_seconds,
            },
        }
        self._semaphore = asyncio.Semaphore(max(1, settings.ai_image_provider_max_concurrency))
        self._http_client = http_client
        self._allowed_output_hosts = frozenset(
            value.strip().lower() for value in settings.openai_image_output_hosts.split(",") if value.strip()
        )
        self._max_output_bytes = settings.ai_image_generated_max_bytes

    # Nhận bối cảnh đã kiểm soát thay vì dữ liệu request thô, giới hạn lời gọi trả phí bằng semaphore dùng chung worker.
    # Hàm chỉ gọi provider một lần cho mỗi output; retry phải được quyết định ở workflow local miễn phí.
    # Lỗi HTTP của OpenAI được phân loại theo status để job chuyển FAILED đúng nguyên nhân an toàn.
    async def generate_lifestyle_background(
        self,
        source: bytes,
        file_name: str,
        request: LifestyleBackgroundRequest,
    ) -> GeneratedImage:
        """Gui bytes source da Media Service kiem soat toi GPT-Image-2 va tra output tam thoi."""

        started_at = time.perf_counter()
        try:
            # Gửi tuple có tên file và MIME rõ ràng; nếu chỉ truyền BytesIO, SDK mặc định octet-stream và OpenAI từ chối.
            # Chuẩn hóa ảnh trong giới hạn profile; profile lifestyle ưu tiên giữ chi tiết sản phẩm hơn tốc độ.
            profile = self._profiles[request.profile]
            prepared_source = await asyncio.to_thread(
                self._prepare_source,
                source,
                int(profile["max_dimension"]),
                int(profile["compression"]),
            )
            image_file = self._build_openai_image_file(prepared_source)
            async with self._semaphore:
                result = await self._client.images.edit(
                    model=self._model,
                    image=image_file,
                    prompt=build_lifestyle_prompt(request),
                    n=1,
                    size=profile["size"],
                    quality=profile["quality"],
                    timeout=float(profile["timeout"]),
                )
            items = result.data or []
            if not items:
                raise ProviderUnavailableError()
            logger.info(
                "OpenAI image generation completed profile=%s duration_ms=%d",
                request.profile.value,
                round((time.perf_counter() - started_at) * 1000),
            )
            item = items[0]
            if getattr(item, "b64_json", None):
                content = self._convert_output_format(
                    base64.b64decode(cast(str, item.b64_json)),
                    str(profile["format"]),
                    int(profile["compression"]),
                )
                return GeneratedImage(
                    content=content,
                    content_type=self._content_type(str(profile["format"])),
                    file_name=f"lifestyle.{profile['format']}",
                    metadata=self._execution_metadata(),
                )
            if getattr(item, "url", None):
                generated_content = await self._download_generated_image(cast(str, item.url))
                generated_content = self._convert_output_format(
                    generated_content,
                    str(profile["format"]),
                    int(profile["compression"]),
                )
                return GeneratedImage(
                    content=generated_content,
                    content_type=self._content_type(str(profile["format"])),
                    file_name=f"lifestyle.{profile['format']}",
                    metadata=self._execution_metadata(),
                )
            raise ProviderUnavailableError()
        except APITimeoutError as error:
            # Timeout là trạng thái provider chậm; không retry vì request tạo ảnh có thể đã được tính phí ở phía OpenAI.
            logger.warning(
                "OpenAI image provider timeout profile=%s duration_ms=%d",
                request.profile.value,
                round((time.perf_counter() - started_at) * 1000),
            )
            raise ProviderTimeoutError() from error
        except OpenAIError as error:
            # Chỉ ghi status/code/type do provider trả về; tuyệt đối không ghi message vì có thể chứa request detail hoặc URL.
            status_code = getattr(error, "status_code", None)
            provider_code = getattr(error, "code", None)
            raw_body: object = getattr(error, "body", None)
            body = raw_body if isinstance(raw_body, dict) else {}
            raw_detail: object = body.get("error")
            detail = raw_detail if isinstance(raw_detail, dict) else {}
            provider_param = detail.get("param") if isinstance(detail.get("param"), str) else None
            provider_type = detail.get("type") if isinstance(detail.get("type"), str) else None
            logger.warning(
                "OpenAI image request rejected type=%s status=%s code=%s param=%s provider_type=%s profile=%s duration_ms=%d",
                type(error).__name__,
                status_code,
                provider_code if isinstance(provider_code, str) else None,
                provider_param,
                provider_type,
                request.profile.value,
                round((time.perf_counter() - started_at) * 1000),
            )
            # 401/403 thường là key chưa bật Image API hoặc tổ chức chưa được xác minh; không coi đây là lỗi mạng tạm thời.
            if status_code in {401, 403}:
                raise ProviderConfigurationError() from error
            # 400/422 là request không tương thích provider; retry sẽ chỉ làm tăng chi phí mà không thể thành công.
            if status_code in {400, 422}:
                raise ProviderRequestRejectedError() from error
            # 429 đến từ OpenAI khác rate limit theo seller của AI Service; không retry tự động lời gọi trả phí.
            if status_code == 429:
                raise ProviderRateLimitedError() from error
            raise ProviderUnavailableError() from error
        except (httpx.HTTPError, TimeoutError, OSError) as error:
            # Lỗi mạng, timeout hoặc dữ liệu ảnh hỏng được giữ ở nhóm unavailable; worker sẽ ghi failure code ổn định.
            logger.warning(
                "OpenAI image provider unavailable type=%s profile=%s duration_ms=%d",
                type(error).__name__,
                request.profile.value,
                round((time.perf_counter() - started_at) * 1000),
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
    def _prepare_source(self, source: bytes, max_dimension: int, jpeg_quality: int) -> bytes:
        """Giảm kích thước và nén ảnh input trước khi gửi tới GPT-Image-2."""

        from PIL import Image, ImageOps

        with Image.open(BytesIO(source)) as image:
            prepared = ImageOps.exif_transpose(image)
            if prepared is None:
                raise OSError("Cannot prepare source image")
            prepared.thumbnail((max(256, max_dimension), max(256, max_dimension)), Image.Resampling.LANCZOS)
            output = BytesIO()
            if "A" in prepared.getbands():
                prepared.convert("RGBA").save(output, format="PNG", optimize=True)
            else:
                prepared.convert("RGB").save(output, format="JPEG", quality=max(1, min(100, jpeg_quality)), optimize=True)
            return output.getvalue()

    # Map output format sang MIME ổn định để Media Service không phải đoán theo tên file.
    @staticmethod
    def _content_type(output_format: str) -> str:
        """Trả MIME hợp lệ cho định dạng ảnh do profile cấu hình."""

        return {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(output_format, "image/jpeg")

    # Chuyển ảnh PNG mặc định của GPT Image về định dạng profile ở local để request provider luôn tối giản, tương thích.
    @staticmethod
    def _convert_output_format(content: bytes, output_format: str, quality: int) -> bytes:
        """Chuẩn hóa bytes ảnh và MIME tương ứng trước khi chuyển cho Media Service."""

        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            output = BytesIO()
            normalized_format = output_format.lower()
            if normalized_format == "jpeg":
                image.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=max(1, min(100, quality)),
                    optimize=True,
                )
            elif normalized_format == "webp":
                image.save(
                    output,
                    format="WEBP",
                    quality=max(1, min(100, quality)),
                    method=6,
                )
            else:
                image.save(output, format="PNG", optimize=True)
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
