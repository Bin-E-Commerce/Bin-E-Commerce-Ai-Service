"""Tạo và cache vector PE-Core của catalog hoặc truy vấn, không lưu bytes ảnh."""

import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.modules.search_ranking.image_embedding.application.image_preprocessing import (
    IMAGE_PREPROCESS_VERSION,
    DetailFraming,
    ImageFraming,
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

logger = logging.getLogger(__name__)


# Cache giữ projection float32 và cờ hình học, không giữ ảnh upload hoặc pixel sau khi trả kết quả.
@dataclass(frozen=True)
class EncodedViews:
    vectors: tuple[bytes, ...]
    has_rotations: bool = False
    strong_match_required: bool = False


# DTO application cho phép router chuyển vector đã kiểm tra sang schema HTTP nội bộ.
@dataclass(frozen=True)
class QueryImageEmbedding:
    vector: list[float]
    detail_vectors: tuple[list[float], list[float]] | None = None
    rotation_vectors: tuple[list[float], list[float]] | None = None
    strong_match_required: bool = False


class ImageEmbeddingProvider(Protocol):
    """Contract model inference; chỉ nhận tensor đã decode và chuẩn hóa."""

    model_version: str
    dimensions: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float
    interpolation: str

    def encode(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]: ...


# Port metric chỉ nhận outcome hữu hạn để theo dõi cache và inference mà không lưu dữ liệu truy vấn.
class ImageEmbeddingCacheMetrics(Protocol):
    """Metric port chỉ nhận label bounded, không chứa ảnh hoặc danh tính request."""

    # Registry tăng counter theo một outcome hữu hạn của cache.
    def increment(self, name: str, labels: dict[str, str] | None = None, amount: int = 1) -> None: ...

    # Registry ghi histogram inference để ước lượng tải CPU mà không giữ từng lần gọi.
    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None: ...


# Use case giữ cache/vector trong RAM và phát metric model local; không lưu ảnh hoặc gọi API tính phí.
class SearchImageEmbeddingService:
    """Validate ảnh, gọi model local và trả vector L2-normalized cùng contract version."""

    def __init__(
        self,
        provider: ImageEmbeddingProvider,
        max_bytes: int,
        max_pixels: int,
        metrics: ImageEmbeddingCacheMetrics | None = None,
    ) -> None:
        """Giữ provider cùng cache RAM bounded; không giữ raw image bytes sau inference."""

        self._provider = provider
        self._max_bytes = max_bytes
        self._max_pixels = max_pixels
        self._metrics = metrics
        self._cache_capacity = 256
        self._cache_ttl_seconds = 600.0
        self._cache: OrderedDict[bytes, tuple[float, EncodedViews]] = OrderedDict()
        self._in_flight: dict[bytes, Future[EncodedViews]] = {}
        self._cache_lock = Lock()

    @property
    def model_version(self) -> str:
        """Expose version từ artifact đã nạp để Recommendation kiểm tra tương thích."""

        return self._provider.model_version

    @property
    def dimensions(self) -> int:
        """Expose chiều vector thật, không suy ra từ cấu hình Qdrant của text."""

        return self._provider.dimensions

    @property
    def preprocess_version(self) -> str:
        """Báo contract preprocessing đã được provider xác nhận từ sidecar model."""

        return IMAGE_PREPROCESS_VERSION

    # Worker catalog tiếp tục decode và tạo tensor theo transform đã dùng để index.
    def embed_image(self, raw: bytes, content_type: str, framing: ImageFraming = "center_crop") -> list[float]:
        # Cache miss mới đọc bytes và chạy inference; caller cùng key dùng lại vector đã xác minh.
        def encode() -> tuple[EncodedViews, bool]:
            image = decode_image(raw, content_type, self._max_bytes, self._max_pixels)
            return EncodedViews((self._encode_view(image, framing),)), True

        vectors = self._cache_vectors(self._cache_key(raw, content_type, framing), encode)
        return np.frombuffer(vectors.vectors[0], dtype=np.float32).tolist()

    # Giải mã một lần và giữ ngân sách tối đa ba inference. Ảnh ngang dùng vùng chi tiết,
    # ảnh dọc dùng hai hướng xoay để Recommendation chọn hướng dựa trên dữ liệu catalog.
    # Chỉ lưu theo hash của query gốc; nhánh phụ bị lỗi vẫn trả vector toàn khung và có thể thử lại.
    def embed_query_image(self, raw: bytes, content_type: str) -> QueryImageEmbedding:
        # Chỉnh góc khung viền phẳng trước khi cắt viền để không mất dấu hai cạnh gốc.
        def encode() -> tuple[EncodedViews, bool]:
            decoded = decode_image(raw, content_type, self._max_bytes, self._max_pixels, preserve_alpha=True)
            image = normalize_query_frame(restore_wide_matte_frame(decoded))
            # Chỉ thử tách ảnh lồng nếu vẫn còn nguyên khung đầu vào.
            if image is decoded:
                image = crop_single_inset_frame(image)
            full = self._encode_view(image, "preserve_frame")
            strong_match_required = requires_strong_match(image, decoded.size)
            primary = EncodedViews((full,), strong_match_required=strong_match_required)
            if not is_elongated_frame(*image.size):
                return primary, True
            try:
                # Hai hướng cần tách thành query vector riêng để không làm trung bình các tư thế khác nhau.
                if image.height > image.width:
                    first = self._encode_view(image.transpose(Image.Transpose.ROTATE_270), "preserve_frame")
                    last = self._encode_view(image.transpose(Image.Transpose.ROTATE_90), "preserve_frame")
                    return EncodedViews((full, first, last), True, strong_match_required), True
                return EncodedViews(
                    (full, self._encode_view(image, "first_detail"), self._encode_view(image, "last_detail")),
                    strong_match_required=strong_match_required,
                ), True
            except Exception as error:
                # Hỏng inference phụ vẫn trả toàn ảnh đã xác minh, không lưu trạng thái suy giảm vào cache.
                logger.warning("Image alternate view embedding unavailable: %s", type(error).__name__)
                return primary, False

        encoded = self._cache_vectors(self._cache_key(raw, content_type, "query_views"), encode)
        rows = [np.frombuffer(vector, dtype=np.float32).tolist() for vector in encoded.vectors]
        auxiliary = (rows[1], rows[2]) if len(rows) == 3 else None
        return QueryImageEmbedding(
            vector=rows[0],
            detail_vectors=None if encoded.has_rotations else auxiliary,
            rotation_vectors=auxiliary if encoded.has_rotations else None,
            strong_match_required=encoded.strong_match_required,
        )

    # Coalesce các request cùng hash. Chỉ giữ float32 đã normalize trong cache, kể cả khi ảnh có nhiều vùng.
    def _cache_vectors(self, cache_key: bytes, encode: Callable[[], tuple[EncodedViews, bool]]) -> EncodedViews:
        now = monotonic()
        with self._cache_lock:
            self._remove_expired_cache_entries(now)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self._record_cache_event("hit")
                return cached[1]
            pending = self._in_flight.get(cache_key)
            if pending is None:
                pending = Future()
                self._in_flight[cache_key] = pending
                is_owner = True
                self._record_cache_event("miss")
            else:
                is_owner = False
                self._record_cache_event("wait")

        if not is_owner:
            # Future chỉ chứa vector float32 đã xác minh từ owner cùng cache key.
            return pending.result()

        try:
            vectors, cacheable = encode()
        except Exception as error:
            # Không cache lỗi decode/model; waiter cùng key nhận lỗi rồi request sau có thể thử lại.
            with self._cache_lock:
                self._in_flight.pop(cache_key, None)
                pending.set_exception(error)
            raise

        with self._cache_lock:
            # Một mục cache tương ứng tối đa ba vector; TTL và số mục vẫn hữu hạn.
            if cacheable:
                self._cache[cache_key] = monotonic() + self._cache_ttl_seconds, vectors
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self._cache_capacity:
                    self._cache.popitem(last=False)
            self._in_flight.pop(cache_key, None)
            pending.set_result(vectors)
        return vectors

    # Phân biệt mọi cách tạo khung trong cache để kết quả catalog không lẫn với truy vấn.
    def _cache_key(self, raw: bytes, content_type: str, framing: str) -> bytes:
        digest = sha256()
        for part in (
            self._provider.model_version,
            self.preprocess_version,
            content_type,
            framing,
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        digest.update(raw)
        return digest.digest()

    # Dọn entry hết hạn trước lookup; dung lượng nhỏ nên quét toàn cache có giới hạn là đủ.
    def _remove_expired_cache_entries(self, now: float) -> None:
        expired_keys = [key for key, (expires_at, _) in self._cache.items() if expires_at <= now]
        for key in expired_keys:
            self._cache.pop(key, None)

    # Chỉ phát metric có ba outcome cố định để cardinality không tăng theo request.
    def _record_cache_event(self, outcome: str) -> None:
        if self._metrics is not None:
            self._metrics.increment(
                "ai_image_embedding_cache_total",
                {"source": "image_embedding", "variant": outcome},
            )

    # Ghi số lần inference thành công/thất bại và thời gian xử lý để vận hành ước lượng tải model local.
    def _record_inference_event(self, outcome: str, duration_seconds: float) -> None:
        if self._metrics is None:
            return
        self._metrics.increment(
            "ai_image_embedding_inference_total",
            {"source": "image_embedding", "variant": outcome},
        )
        self._metrics.observe_histogram(
            "ai_image_embedding_inference_duration_seconds",
            duration_seconds,
        )

    # Mỗi khung dùng tensor cùng chuẩn hóa PE-Core; chuyển thành bytes float32 để cache dữ liệu bất biến.
    def _encode_view(self, image: Image.Image, framing: ImageFraming | DetailFraming) -> bytes:
        tensor = prepare_image_tensor(
            image,
            image_size=self._provider.image_size,
            mean=self._provider.mean,
            std=self._provider.std,
            crop_pct=self._provider.crop_pct,
            interpolation=self._provider.interpolation,
            framing=framing,
        )
        inference_started_at = monotonic()
        try:
            # Timer bắt đầu sau decode/preprocess nên metric chỉ phản ánh model và kiểm tra vector đầu ra.
            output = np.asarray(self._provider.encode(tensor), dtype=np.float32).reshape(-1)
            if output.size != self._provider.dimensions or not np.isfinite(output).all():
                raise RuntimeError("IMAGE_EMBEDDING_OUTPUT_SCHEMA_MISMATCH")
            norm = float(np.linalg.norm(output))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError("IMAGE_EMBEDDING_ZERO_VECTOR")
            # NumPy lưu từng phần tử dưới float32; API application chỉ xuất Python float đã normalize.
            normalized = (output / norm).astype(np.float32).tobytes()
        except Exception:
            # Lỗi từ model hoặc schema vector được tính là inference failure; lỗi ảnh đã bị loại ở bước trước.
            self._record_inference_event("failure", monotonic() - inference_started_at)
            raise
        self._record_inference_event("success", monotonic() - inference_started_at)
        return normalized
