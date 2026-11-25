"""Kiểm thử validate ảnh và metric PE-Core; không gọi ONNX artifact hoặc dịch vụ ngoài."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from threading import Barrier
from time import sleep

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.core.metrics import MetricsRegistry
from app.modules.search_ranking.image_embedding.application import image_preprocessing
from app.modules.search_ranking.image_embedding.application import service as service_module
from app.modules.search_ranking.image_embedding.application.image_preprocessing import (
    InvalidImageData,
    decode_image,
    is_elongated_frame,
    normalize_query_frame,
    prepare_image_tensor,
)
from app.modules.search_ranking.image_embedding.application.query_geometry import (
    crop_single_inset_frame,
    requires_strong_match,
    restore_wide_matte_frame,
)
from app.modules.search_ranking.image_embedding.application.service import SearchImageEmbeddingService


# Tạo ảnh fixture trong memory để test không ghi ảnh người dùng hoặc file tạm ra đĩa.
def _make_png(
    width: int = 32,
    height: int = 24,
    color: tuple[int, int, int] = (120, 80, 40),
) -> bytes:
    """Tạo PNG fixture trong RAM để kiểm tra cache mà không lưu ảnh ra disk."""

    buffer = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


# Provider giả cô lập test khỏi ONNX artifact và kiểm tra đầu vào đúng tensor contract.
class FakeImageEmbeddingProvider:
    model_version = "pe-core-test-v1"
    dimensions = 3
    image_size = 16
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    crop_pct = 1.0
    interpolation = "bicubic"

    def __init__(self) -> None:
        self.encode_calls = 0

    # Trả projection xác định để test normalization mà không cần tải model hoặc gọi mạng.
    def encode(self, tensor: np.ndarray) -> np.ndarray:
        assert tensor.shape == (1, 3, 16, 16)
        self.encode_calls += 1
        return np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32)


# Ảnh hợp lệ được chuyển thành tensor NCHW đúng kích thước và kiểu số model yêu cầu.
def test_prepare_image_tensor_normalizes_shape_and_dtype() -> None:
    tensor = prepare_image_tensor(
        decode_image(_make_png(), "image/png", 1024 * 1024, 1000),
        image_size=16,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
        interpolation="bicubic",
    )

    assert tensor.shape == (1, 3, 16, 16)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


# Đường resize có giới hạn giữ khung cắt cũ mà chỉ cấp phát kích thước đầu vào của model.
@pytest.mark.parametrize("size", [(96, 18), (18, 96)])
def test_bounded_center_crop_matches_direct_resize(size: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    width, height = size
    pixels = np.random.default_rng(42).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    prepare = partial(
        prepare_image_tensor,
        image_size=16,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        crop_pct=0.875,
        interpolation="bicubic",
    )
    expected = prepare(image)
    # Hạ giới hạn trên fixture nhỏ để kiểm tra nhánh mới mà không tạo ảnh lớn trong unit test.
    monkeypatch.setattr(image_preprocessing, "_MAX_CENTER_CROP_INTERMEDIATE_PIXELS", 256)
    original_resize = Image.Image.resize
    sizes: list[tuple[int, int]] = []

    # Ghi lại kích thước thực sự được cấp phát, rồi giữ hành vi resize thật để so pixel.
    def track_resize(self: Image.Image, target: tuple[int, int], *args, **kwargs) -> Image.Image:
        sizes.append(target)
        return original_resize(self, target, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", track_resize)
    np.testing.assert_allclose(prepare(image), expected, atol=1 / 255)
    assert sizes == [(16, 16)]


# Hai đầu ảnh dọc hoặc ngang đều phải còn trong tensor query sau khi thêm viền.
@pytest.mark.parametrize("vertical", [False, True])
def test_preserve_frame_keeps_both_ends_of_elongated_query(vertical: bool) -> None:
    # Đặt hai mảng màu ở đầu ảnh để phát hiện vùng bị mất khi cắt giữa.
    image = Image.new("RGB", (48, 16), color=(0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 12, 16))
    image.paste((0, 255, 0), (36, 0, 48, 16))
    if vertical:
        image = image.transpose(Image.Transpose.TRANSPOSE)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    prepare = partial(
        prepare_image_tensor,
        decode_image(buffer.getvalue(), "image/png", 1024 * 1024, 1000),
        image_size=48,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        crop_pct=1.0,
        interpolation="bicubic",
    )

    catalog = prepare()
    query = prepare(framing="preserve_frame")

    start = (2, 24) if vertical else (24, 2)
    end = (45, 24) if vertical else (24, 45)
    assert catalog.max() == 0.0
    assert query[0, :, *start].tolist() == [1.0, 0.0, 0.0]
    assert query[0, :, *end].tolist() == [0.0, 1.0, 0.0]
    assert query[0, :, 0, 0].tolist() == pytest.approx([127 / 255] * 3)
    # Mỗi vùng chỉ thấy đầu tương ứng, kể cả khi ảnh dựng dọc.
    assert prepare(framing="first_detail")[0, 1].max() == 0.0
    assert prepare(framing="last_detail")[0, 0].max() == 0.0
    assert prepare(framing="first_detail")[0, 0].max() == 1.0
    assert prepare(framing="last_detail")[0, 1].max() == 1.0


# MIME khai báo không đủ để chấp nhận payload; magic bytes phải xác nhận nội dung ảnh.
def test_decode_image_rejects_mismatched_magic_bytes() -> None:
    with pytest.raises(InvalidImageData, match="IMAGE_TYPE_OR_MAGIC_UNSUPPORTED"):
        decode_image(b"not an image", "image/png", max_bytes=1024, max_pixels=1000)


# Pixel cap được áp dụng trước khi Pillow giải nén dữ liệu để giảm rủi ro decompression bomb.
def test_decode_image_rejects_oversized_pixel_count() -> None:
    with pytest.raises(InvalidImageData, match="IMAGE_PIXEL_LIMIT_EXCEEDED"):
        decode_image(_make_png(width=40, height=40), "image/png", max_bytes=1024, max_pixels=1000)


# Application chỉ trả vector hữu hạn đã normalize; model output không được chuyển thẳng ra Qdrant.
def test_search_embedding_service_l2_normalizes_provider_output() -> None:
    service = SearchImageEmbeddingService(FakeImageEmbeddingProvider(), 1024 * 1024, 1000)

    vector = service.embed_image(_make_png(), "image/png")

    assert vector == pytest.approx([0.6, 0.8, 0.0])
    assert np.linalg.norm(vector) == pytest.approx(1.0)


# Output sai dimension bị fail closed thay vì pad hoặc truncate sang vector space khác.
def test_search_embedding_service_rejects_wrong_output_dimension() -> None:
    provider = FakeImageEmbeddingProvider()
    provider.encode = lambda _: np.asarray([[1.0, 2.0]], dtype=np.float32)  # type: ignore[method-assign]
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)

    with pytest.raises(RuntimeError, match="IMAGE_EMBEDDING_OUTPUT_SCHEMA_MISMATCH"):
        service.embed_image(_make_png(), "image/png")


# Cache chỉ giữ vector đã chuẩn hóa; metric phải tách cache hit khỏi inference ONNX thật.
def test_search_embedding_service_caches_successful_vector() -> None:
    provider = FakeImageEmbeddingProvider()
    metrics = MetricsRegistry()
    service = SearchImageEmbeddingService(
        provider,
        1024 * 1024,
        1000,
        metrics=metrics,
    )
    image = _make_png()

    first = service.embed_image(image, "image/png")
    second = service.embed_image(image, "image/png")

    assert first == second == pytest.approx([0.6, 0.8, 0.0])
    assert provider.encode_calls == 1
    rendered = metrics.render()
    assert 'ai_image_embedding_cache_total{source="image_embedding",variant="hit"} 1' in rendered
    assert 'ai_image_embedding_cache_total{source="image_embedding",variant="miss"} 1' in rendered
    assert 'ai_image_embedding_inference_total{source="image_embedding",variant="success"} 1' in rendered
    assert "ai_image_embedding_inference_duration_seconds_count 1" in rendered


# Cùng bytes nhưng vector catalog và bộ vector truy vấn phải tách cache vì cách tạo khung khác nhau.
def test_search_embedding_cache_separates_catalog_and_query_framing() -> None:
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)
    image = _make_png(width=48, height=16)

    for _ in range(2):
        service.embed_image(image, "image/png")
        service.embed_query_image(image, "image/png")

    assert provider.encode_calls == 4


# Ảnh vuông không cần vùng; ảnh viền vuông có nội dung dài chỉ decode một lần và dùng lại bộ ba vector.
def test_detail_views_are_bounded_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 30_000)
    square = _make_png(width=16, height=16)
    image = Image.new("RGB", (160, 160), color=(242, 242, 242))
    image.paste((50, 80, 120), (20, 55, 142, 108))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    decode_calls = 0
    decode_image = service_module.decode_image

    # Đếm decoder thật để phát hiện việc tách inference rồi vô tình decode lại cho mỗi vùng.
    def count_decode(
        raw: bytes, content_type: str, max_bytes: int, max_pixels: int, *, preserve_alpha: bool = False
    ) -> Image.Image:
        nonlocal decode_calls
        decode_calls += 1
        return decode_image(raw, content_type, max_bytes, max_pixels, preserve_alpha=preserve_alpha)

    monkeypatch.setattr(service_module, "decode_image", count_decode)

    assert service.embed_query_image(square, "image/png").detail_vectors is None
    first = service.embed_query_image(buffer.getvalue(), "image/png")
    assert first.detail_vectors is not None
    assert first == service.embed_query_image(buffer.getvalue(), "image/png")
    assert provider.encode_calls == 4
    assert decode_calls == 2


# Ảnh dọc nhận hai góc 90° trong cùng ngân sách ba inference, tách khỏi vùng chi tiết của ảnh ngang.
def test_portrait_query_exposes_two_orientation_vectors() -> None:
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)

    result = service.embed_query_image(_make_png(width=16, height=48), "image/png")

    assert result.rotation_vectors is not None
    assert result.detail_vectors is None
    assert provider.encode_calls == 3


# Khi viền phẳng làm khung ngang nghiêng trông gần vuông, khôi phục hình bao và hai vùng của ảnh gốc.
@pytest.mark.parametrize("tilt", [16, -16])
def test_angled_wide_photo_restores_detail_views(tilt: int) -> None:
    background = (247, 247, 247)
    photo = Image.new("RGB", (420, 200), (210, 215, 225))
    photo.paste((100, 65, 45), (35, 50, 385, 150))
    uploaded = photo.rotate(tilt, Image.Resampling.BICUBIC, expand=True, fillcolor=background)
    buffer = BytesIO()
    uploaded.save(buffer, format="PNG")
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 200_000)

    restored = normalize_query_frame(restore_wide_matte_frame(uploaded))
    embedding = service.embed_query_image(buffer.getvalue(), "image/png")

    assert not is_elongated_frame(*uploaded.size)
    assert restored.width / restored.height == pytest.approx(2.1, abs=0.12)
    assert embedding.detail_vectors is not None
    assert embedding.rotation_vectors is None
    assert provider.encode_calls == 3


# Vật thể cong có hình bao dài nhưng không có hai cạnh đối diện thẳng, nên giữ góc người dùng tải lên.
def test_angled_oval_keeps_uploaded_orientation() -> None:
    background = (247, 247, 247)
    image = Image.new("RGB", (420, 200), background)
    ImageDraw.Draw(image).ellipse((9, 4, 411, 196), fill=(76, 94, 127))
    uploaded = image.rotate(16, Image.Resampling.BICUBIC, expand=True, fillcolor=background)

    assert restore_wide_matte_frame(uploaded) is uploaded


# Quá ít pixel hoặc tiền cảnh cắt qua khung đáng kể cần điểm khớp cao hơn; ảnh có đủ lề dùng ngưỡng thường.
def test_strong_match_flag_tracks_small_and_clipped_inputs() -> None:
    cropped = Image.new("RGB", (300, 200), (237, 237, 237))
    cropped.paste((38, 54, 64), (0, 30, 260, 173))
    centered = Image.new("RGB", (300, 200), (237, 237, 237))
    centered.paste((38, 54, 64), (20, 30, 260, 173))

    assert requires_strong_match(Image.new("RGB", (160, 76)), (160, 76))
    assert requires_strong_match(cropped, cropped.size)
    assert not requires_strong_match(centered, centered.size)
    assert not requires_strong_match(Image.new("RGB", (90, 90)), (90, 90))


# Một ảnh lồng có bốn mép được giữ lại; nền có đường kẻ chạy xuyên không được nhận nhầm là ảnh lồng.
def test_inset_detection_requires_a_closed_frame() -> None:
    background = Image.new("RGB", (640, 640), (224, 217, 202))
    draw = ImageDraw.Draw(background)
    for coordinate in range(0, 640, 38):
        draw.line((coordinate, 0, coordinate, 640), fill=(172, 158, 135), width=3)
        draw.line((0, coordinate, 640, coordinate), fill=(189, 176, 158), width=2)
    framed = background.copy()
    framed.paste((252, 252, 252), (188, 92, 593, 291))
    framed.paste((68, 78, 89), (211, 144, 554, 261))

    actual = crop_single_inset_frame(framed)

    assert crop_single_inset_frame(background) is background
    assert 389 <= actual.width <= 405
    assert 183 <= actual.height <= 199
    assert actual.getpixel((actual.width // 2, actual.height // 2)) == (68, 78, 89)


# Màu RGB trong pixel alpha 0 không được đưa vào model dù file còn giữ màu ẩn.
def test_query_normalization_discards_hidden_alpha_colors() -> None:
    image = Image.new("RGBA", (72, 50), color=(12, 200, 50, 0))
    image.paste((100, 60, 10, 255), (10, 9, 65, 38))
    image.putpixel((30, 20), (0, 130, 250, 0))

    normalized = normalize_query_frame(image)

    assert normalized.mode == "RGB"
    assert normalized.size == (55, 29)
    assert normalized.getpixel((20, 11)) == (242, 242, 242)


# Nền nhiều màu không đủ bằng chứng là viền nhân tạo, cần giữ toàn khung của ảnh.
def test_query_normalization_preserves_nonuniform_background() -> None:
    image = Image.new("RGB", (160, 160), color=(242, 242, 242))
    image.paste((100, 60, 10), (40, 40, 120, 120))
    image.paste((200, 208, 217), (0, 0, 4, 4))

    assert normalize_query_frame(image) is image


# Các ảnh khác nhau không được gộp vào một cache entry dù cùng model và cùng MIME.
def test_search_embedding_service_cache_key_uses_image_bytes() -> None:
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)

    service.embed_image(_make_png(color=(1, 2, 3)), "image/png")
    service.embed_image(_make_png(color=(4, 5, 6)), "image/png")

    assert provider.encode_calls == 2


# MIME là một phần của key; ảnh MIME sai phải bị từ chối trước model và không tính inference failure.
def test_search_embedding_service_cache_key_uses_content_type() -> None:
    provider = FakeImageEmbeddingProvider()
    metrics = MetricsRegistry()
    service = SearchImageEmbeddingService(
        provider,
        1024 * 1024,
        1000,
        metrics=metrics,
    )
    image = _make_png()
    service.embed_image(image, "image/png")

    with pytest.raises(InvalidImageData):
        service.embed_image(image, "image/jpeg")

    assert provider.encode_calls == 1
    rendered = metrics.render()
    assert 'ai_image_embedding_inference_total{source="image_embedding",variant="success"} 1' in rendered
    assert 'ai_image_embedding_inference_total{source="image_embedding",variant="failure"}' not in rendered
    assert "ai_image_embedding_inference_duration_seconds_count 1" in rendered


# Cache hết hạn sau TTL; request mới chạy validation/inference như cache miss bình thường.
def test_search_embedding_service_expires_cached_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.modules.search_ranking.image_embedding.application.service as service_module

    now = [100.0]
    monkeypatch.setattr(service_module, "monotonic", lambda: now[0])
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)
    image = _make_png()

    service.embed_image(image, "image/png")
    now[0] += 601
    service.embed_image(image, "image/png")

    assert provider.encode_calls == 2


# Cache eviction giữ tối đa 256 vector trong mỗi process dù query ảnh tiếp tục thay đổi.
def test_search_embedding_service_bounds_cache_entries() -> None:
    provider = FakeImageEmbeddingProvider()
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)

    for index in range(257):
        service.embed_image(
            _make_png(color=(index % 256, index // 256, 0)),
            "image/png",
        )

    assert len(service._cache) == 256
    assert provider.encode_calls == 257


# Request đồng thời cùng ảnh dùng chung một future để không nhân đôi inference ONNX.
def test_search_embedding_service_single_flights_same_image() -> None:
    provider = FakeImageEmbeddingProvider()
    original_encode = provider.encode

    def slow_encode(tensor: np.ndarray) -> np.ndarray:
        # Làm chậm inference giả để các waiter cùng quan sát một future đang chạy.
        sleep(0.05)
        return original_encode(tensor)

    provider.encode = slow_encode  # type: ignore[method-assign]
    service = SearchImageEmbeddingService(provider, 1024 * 1024, 1000)
    image = _make_png()
    barrier = Barrier(8)

    def embed_together() -> list[float]:
        """Đồng bộ start để các thread cùng tranh một cache key."""

        barrier.wait()
        return service.embed_image(image, "image/png")

    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(lambda _: embed_together(), range(8)))

    assert all(vector == pytest.approx([0.6, 0.8, 0.0]) for vector in vectors)
    assert provider.encode_calls == 1


# Lỗi inference không được cache; metric đếm cả failure lẫn retry thành công nhưng không lộ exception.
def test_search_embedding_service_does_not_cache_failures() -> None:
    provider = FakeImageEmbeddingProvider()
    call_count = [0]

    # Lần đầu mô phỏng lỗi model, lần sau trả projection hợp lệ để kiểm tra retry.
    def fail_once(tensor: np.ndarray) -> np.ndarray:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("model temporarily failed")
        return np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32)

    provider.encode = fail_once  # type: ignore[method-assign]
    metrics = MetricsRegistry()
    service = SearchImageEmbeddingService(
        provider,
        1024 * 1024,
        1000,
        metrics=metrics,
    )
    image = _make_png()

    with pytest.raises(RuntimeError, match="model temporarily failed"):
        service.embed_image(image, "image/png")
    assert service.embed_image(image, "image/png") == pytest.approx([0.6, 0.8, 0.0])
    assert service.embed_image(image, "image/png") == pytest.approx([0.6, 0.8, 0.0])

    assert call_count[0] == 2
    rendered = metrics.render()
    assert 'ai_image_embedding_inference_total{source="image_embedding",variant="failure"} 1' in rendered
    assert 'ai_image_embedding_inference_total{source="image_embedding",variant="success"} 1' in rendered
    assert "ai_image_embedding_inference_duration_seconds_count 2" in rendered
    assert "model temporarily failed" not in rendered
