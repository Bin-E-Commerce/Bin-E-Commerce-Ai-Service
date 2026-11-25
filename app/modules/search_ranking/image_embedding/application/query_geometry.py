"""Nhận diện khung ảnh lồng và dấu hiệu mất chi tiết từ ảnh truy vấn đã decode."""

import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.modules.search_ranking.image_embedding.application.image_preprocessing import is_elongated_frame


# Khôi phục khung ngang bị nghiêng trên nền viền phẳng khi tỷ lệ file khiến query sắp bị cắt giữa.
# Chỉ đo trên preview và yêu cầu hai cạnh đối diện đủ thẳng, tránh xoay theo vật thể tròn hoặc dáng chụp.
def restore_wide_matte_frame(image: Image.Image) -> Image.Image:
    if image.mode != "RGB" or image.height < 240 or image.width <= image.height or is_elongated_frame(*image.size):
        return image
    scale = 192 / image.width
    preview = image.resize((192, round(image.height * scale)), Image.Resampling.BOX)
    pixels = np.asarray(preview, dtype=np.int16)
    corners = np.stack((pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]))
    if np.ptp(corners, axis=0).max() > 6:
        return image

    background = np.median(corners, axis=0).astype(np.int16)
    mask = np.abs(pixels - background).max(axis=2) > 8
    if mask.mean() < 0.20:
        return image
    rows, columns = np.nonzero(mask)
    initial_area = float((np.ptp(columns) + 1) * (np.ptp(rows) + 1))
    best_area, best_angle = initial_area, 0
    end_axis, edge_axis = columns.astype(np.float64), rows.astype(np.float64)

    # So diện tích hình bao trong khoảng góc nhỏ. Không lấy góc chạm biên vì có thể ảnh quay nhiều hơn phạm vi này.
    for angle in range(-24, 25):
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        x = columns * cosine + rows * sine
        y = -columns * sine + rows * cosine
        area = float((np.ptp(x) + 1) * (np.ptp(y) + 1))
        if area < best_area:
            best_area, best_angle = area, angle
            end_axis, edge_axis = x, y

    if (
        not 5 <= abs(best_angle) < 24
        or best_area > 0.80 * initial_area
        or len(rows) < 0.65 * best_area
        or np.ptp(end_axis) < 1.65 * np.ptp(edge_axis)
        or not _has_parallel_frame_ends(end_axis, edge_axis)
    ):
        return image

    # Giới hạn cạnh đầu vào của phép xoay, vì toàn khung chỉ dùng để tạo tensor PE-Core có kích thước cố định.
    rendered = image
    if image.width > 1280:
        rendered = image.resize((1280, round(image.height * 1280 / image.width)), Image.Resampling.LANCZOS)
    color = int(background[0]), int(background[1]), int(background[2])
    return rendered.rotate(best_angle, Image.Resampling.BICUBIC, expand=True, fillcolor=color)


# Hai cạnh ngắn phải phủ hầu hết chiều cao khung chiếu. Chỉ nhìn kích thước hộp sẽ nhận nhầm vật thể cong.
def _has_parallel_frame_ends(end_axis: NDArray[np.float64], edge_axis: NDArray[np.float64]) -> bool:
    low, high = float(edge_axis.min()), float(edge_axis.max())
    padding = 0.07 * (high - low)
    bins = np.linspace(low + padding, high - padding, 25)
    for edge in (end_axis.min(), end_axis.max()):
        near_edge = np.abs(end_axis - edge) <= 3
        hits, _ = np.histogram(edge_axis[near_edge], bins)
        if np.mean(hits > 0) < 0.8:
            return False
    return True


# Chỉ siết ngưỡng với ảnh rất nhỏ và hẹp hoặc tiền cảnh có một cạnh bị cắt rõ trên nền đồng màu.
# Dấu hiệu này không phân loại đồ vật; ảnh có đủ điểm tương đồng vẫn được tìm bình thường.
def requires_strong_match(image: Image.Image, source_size: tuple[int, int]) -> bool:
    short, long = sorted(source_size)
    # Dùng kích thước upload để ảnh vuông nhỏ không bị đánh giá là quá hẹp do bước bỏ nền.
    if short < 96 and long <= 256 and long > short * 1.5:
        return True
    if image.width < image.height * 1.25:
        return False

    pixels = np.asarray(image.resize((96, 96), Image.Resampling.BOX), dtype=np.int16)
    corners = np.stack((pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]))
    if np.ptp(corners, axis=0).max() > 12:
        return False

    background = np.median(corners, axis=0).astype(np.int16)
    occupied = np.abs(pixels - background).max(axis=2) > 28
    # Một cạnh bị che hơn nửa và cạnh đối diện đa số là nền giúp phân biệt crop với ảnh lấp đều hai phía.
    edges = (float(occupied[:, :2].mean()), float(occupied[:, -2:].mean()))
    return max(edges) >= 0.5 and min(edges) < 0.2


# Chỉ tách ảnh lồng khi thấy một hình chữ nhật nổi bật có đủ bốn cạnh.
# Giới hạn preview và số đường ứng viên; nền kẻ chạy xuyên khung hoặc nhiều hình lồng thì để nguyên.
def crop_single_inset_frame(image: Image.Image) -> Image.Image:
    if min(image.size) < 240 or max(image.size) > 1.25 * min(image.size):
        return image
    scale = 192 / max(image.size)
    width, height = round(image.width * scale), round(image.height * scale)
    preview = np.asarray(image.resize((width, height), Image.Resampling.BOX), dtype=np.int16)
    horizontal = np.abs(preview[2:] - preview[:-2]).max(axis=2) >= 18
    vertical = np.abs(preview[:, 2:] - preview[:, :-2]).max(axis=2) >= 18
    lines: list[tuple[int, int, int]] = []
    for y in range(4, height - 5):
        indices = np.flatnonzero(horizontal[y])
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1):
            # Bỏ các đường ngắn và đường kéo gần hết ảnh, thường là nền kẻ chứ không phải khung lồng.
            if len(run) >= 0.30 * width and run[-1] - run[0] + 1 <= 0.88 * width:
                lines.append((y, int(run[0]), int(run[-1]) + 1))
        if len(lines) > 32:
            return image

    matches: list[tuple[float, tuple[int, int, int, int]]] = []
    for i, (top, left, right) in enumerate(lines):
        for bottom, other_left, other_right in lines[i + 1 :]:
            height_fraction = (bottom - top) / height
            area_fraction = height_fraction * (right - left) / width
            if (
                not 0.19 <= height_fraction <= 0.82
                or not 0.11 <= area_fraction <= 0.72
                or abs(left - other_left) > 4
                or abs(right - other_right) > 4
                or left < 4
                or right > width - 4
            ):
                continue
            # Hai đường ngang cùng độ rộng vẫn phải có hai đường dọc nối khung trên phần lớn chiều cao.
            sides = [float(vertical[top:bottom, x - 3 : x + 2].any(axis=1).mean()) for x in (left, right)]
            ends = [float(horizontal[y, left:right].mean()) for y in (top, bottom)]
            if min(sides) >= 0.65 and min(ends) >= 0.72:
                matches.append((sum(sides) + sum(ends), (left, top + 1, right, bottom + 1)))

    if not matches:
        return image
    score, bounds = max(matches)
    # Nhiều đường gần nhau có thể là cùng một mép. Không tự chọn giữa hai khung khác nhau mà bằng chứng gần ngang nhau.
    if any(
        rival >= score - 0.2 and any(abs(a - b) > 6 for a, b in zip(candidate, bounds, strict=True))
        for rival, candidate in matches
    ):
        return image
    left, top, right, bottom = (round(coordinate / scale) for coordinate in bounds)
    radius = math.ceil(3 / scale)
    # Xác định lại vị trí cạnh trên ảnh gốc. Preview có thể lệch vài pixel làm lẫn họa tiết nền hoặc cắt sát chi tiết.
    new_left = _refine_frame_edge(image, left, top, bottom, radius, vertical=True)
    new_top = _refine_frame_edge(image, top, left, right, radius, vertical=False)
    new_right = _refine_frame_edge(image, right, top, bottom, radius, vertical=True)
    new_bottom = _refine_frame_edge(image, bottom, left, right, radius, vertical=False)
    if new_left is not None and new_top is not None and new_right is not None and new_bottom is not None:
        return image.crop((new_left, new_top, new_right, new_bottom))
    # Nếu không xác định chính xác cả bốn cạnh, lùi vào một pixel preview để tránh lẫn đường nền.
    inset = max(1, round(1 / scale))
    return image.crop((left + inset, top + inset, right - inset, bottom - inset))


# Chỉ đọc một dải hẹp quanh mép dự kiến. Nét kẻ chạy xuyên nền bị trừ điểm vì còn kéo dài ngoài hình lồng.
def _refine_frame_edge(
    image: Image.Image,
    edge: int,
    cross_start: int,
    cross_end: int,
    radius: int,
    *,
    vertical: bool,
) -> int | None:
    parallel, cross = image.size if vertical else (image.height, image.width)
    start, end = max(0, edge - radius), min(parallel, edge + radius + 1)
    if vertical:
        band = image.crop((start, 0, end, cross))
    else:
        band = image.crop((0, start, cross, end)).transpose(Image.Transpose.TRANSPOSE)
    height = min(cross, 192)
    pixels = np.asarray(band.resize((end - start, height), Image.Resampling.BOX), dtype=np.int16)
    gradient = np.abs(np.diff(pixels, axis=1)).max(axis=2)
    begin, finish = round(cross_start * height / cross), round(cross_end * height / cross)
    pad = max(1, (finish - begin) // 10)
    inside = np.median(gradient[begin + pad : finish - pad], axis=0)
    outside = np.concatenate((gradient[: max(0, begin - pad)], gradient[finish + pad :]))
    if not outside.size:
        return None
    contrast = inside - np.median(outside, axis=0)
    best = int(contrast.argmax())
    return start + best + 1 if contrast[best] >= 15 else None
