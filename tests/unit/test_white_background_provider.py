"""Kiểm thử provider nền trắng không tạo false-success khi rembg lỗi.

Test dùng segmentation fake trong RAM, không tải model, không đọc file và không gọi mạng.
"""

from io import BytesIO

import pytest
from PIL import Image

from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.infrastructure.providers.white_background import WhiteBackgroundProvider


# Tạo PNG nhỏ trong bộ nhớ làm source deterministic cho mọi case.
def _source_png() -> bytes:
    """Trả ảnh RGBA opaque để fake segmentation chủ động tạo alpha mask."""

    image = Image.new("RGBA", (32, 32), (220, 40, 40, 255))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# Xác nhận thiếu rembg không được báo thành công bằng ảnh gốc giả tối ưu.
@pytest.mark.asyncio
async def test_rejects_request_when_background_model_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """AAA: provider phải fail rõ ràng khi segmentation dependency không sẵn sàng."""

    provider = WhiteBackgroundProvider(max_dimension=64, webp_quality=88)
    monkeypatch.setattr(provider, "_get_rembg_components", lambda: (None, None))

    with pytest.raises(ProviderUnavailableError):
        await provider.generate_white_background(_source_png(), "product.png")


# Xác nhận pipeline ghép nền trắng khi segmentation trả alpha mask hợp lệ.
@pytest.mark.asyncio
async def test_generates_webp_from_segmented_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    """AAA: fake segmentation tránh tải model nhưng vẫn kiểm tra pipeline output thật."""

    # Fake remove tạo một pixel trong suốt để mô phỏng alpha mask từ rembg.
    def fake_remove(source: bytes, *, session: object) -> bytes:
        """Trả PNG có alpha và giữ nguyên phần còn lại của ảnh nguồn."""

        del session
        with Image.open(BytesIO(source)) as image:
            foreground = image.convert("RGBA")
            foreground.putpixel((0, 0), (0, 0, 0, 0))
            output = BytesIO()
            foreground.save(output, format="PNG")
            return output.getvalue()

    provider = WhiteBackgroundProvider(max_dimension=64, webp_quality=88)
    monkeypatch.setattr(provider, "_get_rembg_components", lambda: (fake_remove, object()))

    result = await provider.generate_white_background(_source_png(), "product.png")

    assert result.content_type == "image/webp"
    assert result.file_name == "product-white.webp"
    with Image.open(BytesIO(result.content)) as output:
        assert output.format == "WEBP"
        assert output.size == (32, 32)
