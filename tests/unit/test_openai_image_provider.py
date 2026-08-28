"""Kiểm tra adapter lifestyle gửi request tương thích và chuẩn hóa ảnh đầu ra."""

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import SecretStr

from app.core.config import Settings
from app.modules.image_optimization.application.ports import LifestyleBackgroundRequest
from app.modules.image_optimization.domain.enums import ImageGenerationProfile, LifestyleBackgroundPreset
from app.modules.image_optimization.infrastructure.providers import openai_image


def _source_png() -> bytes:
    """Tạo source nhỏ để test không đọc ảnh thật hoặc gọi network."""

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_lifestyle_request_uses_conservative_edit_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request chỉ dùng tham số được API edit hỗ trợ ổn định và convert PNG về JPEG local."""

    calls: list[dict[str, object]] = []
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "black").save(output, format="PNG")
    encoded_output = base64.b64encode(output.getvalue()).decode("ascii")

    class FakeImages:
        async def edit(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded_output, url=None)])

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.images = FakeImages()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai_image, "AsyncOpenAI", FakeClient)
    provider = openai_image.OpenAILifestyleImageProvider(
        Settings(openai_api_key=SecretStr("test-key"), ai_image_preview_format="jpeg")
    )

    result = await provider.generate_lifestyle_background(
        _source_png(),
        "source.png",
        LifestyleBackgroundRequest(
            preset=LifestyleBackgroundPreset.WARM_HOME,
            description=None,
            profile=ImageGenerationProfile.PREVIEW,
        ),
    )

    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-image-2"
    assert calls[0]["size"] == "1024x1024"
    assert calls[0]["quality"] == "low"
    assert "input_fidelity" not in calls[0]
    assert "output_format" not in calls[0]
    assert "output_compression" not in calls[0]
    assert "background" not in calls[0]
    assert result.content_type == "image/jpeg"
    assert result.file_name == "lifestyle.jpeg"
    with Image.open(io.BytesIO(result.content)) as image:
        assert image.format == "JPEG"
