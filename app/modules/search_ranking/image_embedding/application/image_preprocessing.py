"""Decode ảnh an toàn và chuẩn hóa tensor PE-Core, không lưu ảnh hoặc vector."""

import math
import warnings
from io import BytesIO
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

_IMAGE_FORMATS = {
    "image/jpeg": ("JPEG", b"\xff\xd8\xff"),
    "image/png": ("PNG", b"\x89PNG\r\n\x1a\n"),
}
_WEBP_MAGIC = b"RIFF"
IMAGE_PREPROCESS_VERSION = "pe-core-image-transform-v1"
ImageFraming = Literal["center_crop", "preserve_frame"]
DetailFraming = Literal["first_detail", "last_detail"]
_MAX_QUERY_CROP_ASPECT_RATIO = 1.5
_MAX_CENTER_CROP_INTERMEDIATE_PIXELS = 4_000_000
_QUERY_PADDING_PREVIEW_EDGE = 640
_QUERY_PADDING_COLOR_TOLERANCE = 3
_MAX_QUERY_CONTENT_FRACTION_FOR_TRIM = 0.82


class InvalidImageData(ValueError):
    """Ảnh không thuộc định dạng, giới hạn hoặc cấu trúc được phép xử lý."""


# Dùng cùng ranh giới cho toàn khung và hai vùng chi tiết để giữ hành vi với ảnh gần vuông.
def is_elongated_frame(width: int, height: int) -> bool:
    return max(width, height) > _MAX_QUERY_CROP_ASPECT_RATIO * min(width, height)


# Chỉ cho phép format có magic bytes xác định; parser Pillow sau đó xác nhận nội dung thực.
def _expected_image_format(content_type: str, raw: bytes) -> str:
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type == "image/webp":
        if len(raw) < 12 or raw[:4] != _WEBP_MAGIC or raw[8:12] != b"WEBP":
            raise InvalidImageData("IMAGE_MAGIC_MISMATCH")
        return "WEBP"
    expected = _IMAGE_FORMATS.get(normalized_type)
    if expected is None or not raw.startswith(expected[1]):
        raise InvalidImageData("IMAGE_TYPE_OR_MAGIC_UNSUPPORTED")
    return expected[0]


# Chặn kích thước/pixel/ảnh động trước khi decode. Chỉ giữ alpha nếu caller là truy vấn ảnh.
def _decode_image(raw: bytes, expected_format: str, max_pixels: int, *, preserve_alpha: bool) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as candidate:
                if candidate.format != expected_format:
                    raise InvalidImageData("IMAGE_FORMAT_MISMATCH")
                if getattr(candidate, "n_frames", 1) != 1:
                    raise InvalidImageData("ANIMATED_IMAGE_UNSUPPORTED")
                if candidate.width < 1 or candidate.height < 1 or candidate.width * candidate.height > max_pixels:
                    raise InvalidImageData("IMAGE_PIXEL_LIMIT_EXCEEDED")
                candidate.verify()
            with Image.open(BytesIO(raw)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImageData("ANIMATED_IMAGE_UNSUPPORTED")
                image.load()
                transposed = ImageOps.exif_transpose(image)
                # Pillow có thể khai báo kết quả transpose là None; từ chối thay vì bỏ qua orientation hoặc trả ảnh rỗng.
                if transposed is None:
                    raise InvalidImageData("IMAGE_DECODE_FAILED")
                if preserve_alpha and ("A" in transposed.getbands() or "transparency" in transposed.info):
                    return transposed.convert("RGBA")
                return transposed.convert("RGB")
    except InvalidImageData:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        OverflowError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise InvalidImageData("IMAGE_DECODE_FAILED") from error


# Xác thực giới hạn bytes và signature trên mọi đường đi trước khi cấp phát bộ nhớ để decode.
def decode_image(raw: bytes, content_type: str, max_bytes: int, max_pixels: int, *, preserve_alpha: bool = False) -> Image.Image:
    if not raw or len(raw) > max_bytes:
        raise InvalidImageData("IMAGE_BYTE_LIMIT_EXCEEDED")
    expected_format = _expected_image_format(content_type, raw)
    return _decode_image(raw, expected_format, max_pixels, preserve_alpha=preserve_alpha)


# Ước lượng viền editor/ảnh chụp màn hình khi bốn góc gần như đồng màu.
# Dùng ảnh nhỏ và yêu cầu nhiều pixel lệch màu trên mỗi hàng/cột để đốm nhiễu không quyết định khung.
# Chỉ giữ crop nếu bớt đủ diện tích và nội dung còn đủ lớn. Ảnh nền phức tạp giữ khung đã decode.
def normalize_query_frame(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        # Viền alpha không chứa thông tin thị giác; nền trung tính ngăn RGB ẩn xuất hiện trong vector.
        alpha_bounds = image.getchannel("A").getbbox()
        if alpha_bounds is not None:
            image = image.crop(alpha_bounds)
        background = Image.new("RGBA", image.size, (242, 242, 242, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")

    # Thumbnail có ít pixel nên bỏ bước suy đoán viền màu để giữ thêm bối cảnh khi phóng ảnh.
    if max(image.size) <= 128:
        return image

    # Giữ tọa độ gốc cho ảnh thông dụng; preview giúp phí dò viền của ảnh lớn có giới hạn.
    preview = image
    if max(image.size) > _QUERY_PADDING_PREVIEW_EDGE:
        scale = _QUERY_PADDING_PREVIEW_EDGE / max(image.size)
        preview_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        preview = image.resize(preview_size, Image.Resampling.BOX)
    width, height = preview.size
    if min(width, height) < 12:
        return image
    pixels = np.asarray(preview, dtype=np.int16)
    corners = np.stack((pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]))
    if np.ptp(corners, axis=0).max() > _QUERY_PADDING_COLOR_TOLERANCE:
        return image

    # Ngưỡng màu chặt giữ mép ảnh đã được dán vào nền, kể cả khi bản thân ảnh có nền sáng.
    background_color = np.median(corners, axis=0).astype(np.int16)
    content = np.abs(pixels - background_color).max(axis=2) > _QUERY_PADDING_COLOR_TOLERANCE
    rows = np.flatnonzero(content.sum(axis=1) >= max(2, width // 128))
    columns = np.flatnonzero(content.sum(axis=0) >= max(2, height // 128))
    if not rows.size or not columns.size:
        return image
    left, top, right, bottom = int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1
    content_width, content_height = right - left, bottom - top
    if (
        min(content_width, content_height) < 12
        or content_width * content_height > _MAX_QUERY_CONTENT_FRACTION_FOR_TRIM * width * height
    ):
        return image

    # Khi đã thu nhỏ để đo, nới một pixel preview mỗi cạnh để giảm rủi ro cắt mất chi tiết thật.
    if preview is not image:
        left, top = max(0, left - 1), max(0, top - 1)
        right, bottom = min(width, right + 1), min(height, bottom + 1)
    return image.crop(
        (
            left * image.width // width,
            top * image.height // height,
            math.ceil(right * image.width / width),
            math.ceil(bottom * image.height / height),
        )
    )


# Ảnh catalog và ảnh truy vấn gần vuông giữ cách cắt theo sidecar của model.
# Khi ảnh truy vấn có tỷ lệ cạnh vượt 1,5, giữ cả khung để giảm mất đặc điểm ở hai đầu.
# Resize trước khi thêm viền để canvas luôn có đúng kích thước đầu vào của model.
def _frame_image(
    image: Image.Image,
    image_size: int,
    crop_pct: float,
    interpolation: str,
    framing: ImageFraming | DetailFraming,
) -> Image.Image:
    if not math.isfinite(crop_pct) or crop_pct <= 0 or crop_pct > 1:
        raise InvalidImageData("IMAGE_PREPROCESS_CONFIG_INVALID")
    width, height = image.size
    resampling = {
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(interpolation.lower())
    if resampling is None:
        raise InvalidImageData("IMAGE_PREPROCESS_CONFIG_INVALID")

    if framing == "first_detail" or framing == "last_detail":
        # Cắt hai đầu theo cạnh dài sau EXIF transpose; xử lý cả ảnh ngang và ảnh dọc.
        side = min(width, height)
        left = width - side if framing == "last_detail" else 0
        top = height - side if framing == "last_detail" else 0
        square = image.crop((left, top, left + side, top + side))
        return _frame_image(square, image_size, crop_pct, interpolation, "center_crop")

    if framing == "preserve_frame" and is_elongated_frame(width, height):
        # Giữ tỷ lệ gốc và bảo đảm cạnh ngắn có ít nhất một pixel, kể cả ảnh rất dài hoặc rất hẹp.
        scale = image_size / max(width, height)
        resized_image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resampling)
        frame = Image.new("RGB", (image_size, image_size), (127, 127, 127))
        frame.paste(resized_image, ((image_size - resized_image.width) // 2, (image_size - resized_image.height) // 2))
        return frame

    # Giữ nguyên phép resize cạnh ngắn để truy vấn gần vuông và vector catalog tiếp tục dùng transform cũ.
    target = int(image_size / crop_pct)
    if width <= height:
        resized = (target, max(target, int(target * height / width)))
    else:
        resized = (max(target, int(target * width / height)), target)
    left = (resized[0] - image_size) // 2
    top = (resized[1] - image_size) // 2
    if resized[0] * resized[1] > _MAX_CENTER_CROP_INTERMEDIATE_PIXELS:
        # Chiếu khung cắt đích về tọa độ phân số trên ảnh gốc. Pillow chỉ resize ô cần dùng
        # để giữ tỷ lệ của phép cắt giữa mà không cấp phát canvas lớn cho ảnh rất dài.
        width_scale = width / resized[0]
        height_scale = height / resized[1]
        return image.resize(
            (image_size, image_size),
            resampling,
            box=(
                left * width_scale,
                top * height_scale,
                (left + image_size) * width_scale,
                (top + image_size) * height_scale,
            ),
        )
    resized_image = image.resize(resized, resampling)
    return resized_image.crop((left, top, left + image_size, top + image_size))


# Cùng transform tensor cho catalog và truy vấn, có thể tái dùng ảnh đã decode cho nhiều khung.
def prepare_image_tensor(
    image: Image.Image,
    *,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_pct: float,
    interpolation: str,
    framing: ImageFraming | DetailFraming = "center_crop",
) -> NDArray[np.float32]:
    framed = _frame_image(image, image_size, crop_pct, interpolation, framing)
    pixels = np.asarray(framed, dtype=np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    if std_array.min() <= 0 or not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise InvalidImageData("IMAGE_PREPROCESS_CONFIG_INVALID")
    normalized = (pixels - mean_array) / std_array
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...], dtype=np.float32)
