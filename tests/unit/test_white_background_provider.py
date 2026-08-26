"""Kiểm thử provider nền trắng local mà không tải model rembg hay gọi mạng.

Test này bảo vệ contract output WebP và đường xử lý async-to-thread; provider thật vẫn được
warm model trong worker production, còn unit test luôn dùng component rembg giả để ổn định.
"""

from io import BytesIO

import pytest
from PIL import Image

from app.modules.image_optimization.infrastructure.providers.white_background import WhiteBackgroundProvider


# Tạo PNG nhỏ trong bộ nhớ để kiểm tra pipeline mà không đọc file hoặc asset thật.
def _source_png() -> bytes:
    image = Image.new("RGBA", (32, 32), (220, 40, 40, 255))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# Xác nhận provider trả WebP nền trắng và không cần rembg model cho đường fallback local.
@pytest.mark.asyncio
async def test_generates_webp_without_blocking_provider_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = WhiteBackgroundProvider(max_dimension=64, webp_quality=88)
    monkeypatch.setattr(provider, "_get_rembg_components", lambda: (None, None))

    result = await provider.generate_white_background(_source_png(), "product.png")

    assert result.content_type == "image/webp"
    assert result.file_name == "product-white.webp"
    with Image.open(BytesIO(result.content)) as output:
        assert output.format == "WEBP"
        assert output.size == (32, 32)
