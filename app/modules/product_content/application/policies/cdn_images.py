"""Xác minh URL ảnh CDN tại application boundary để mọi transport đều an toàn."""

from collections.abc import Iterable
from urllib.parse import urlparse

from app.core.errors import InvalidInputError
from app.modules.product_content.application.commands import ImageCommand


# So khớp origin chính xác, không dùng startswith có thể bị host kẻ tấn công giả mạo.
def validate_cdn_images(images: Iterable[ImageCommand], configured_origins: str) -> None:
    """Chỉ chấp nhận HTTPS URL thuộc allow-list cấu hình."""

    allowed = {value.strip().rstrip("/") for value in configured_origins.split(",") if value.strip()}
    if not allowed:
        raise InvalidInputError()
    for image in images:
        parsed = urlparse(image.public_url)
        origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        if parsed.scheme != "https" or origin not in allowed:
            raise InvalidInputError()
