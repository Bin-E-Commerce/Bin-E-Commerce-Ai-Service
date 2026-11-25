"""Nạp artifact PE-Core ONNX đã pin contract, không tải model lúc startup hoặc request."""

import hashlib
import json
import logging
from hmac import compare_digest
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from app.core.config import Settings
from app.modules.search_ranking.image_embedding.application.image_preprocessing import (
    IMAGE_PREPROCESS_VERSION,
)

logger = logging.getLogger(__name__)
IMAGE_EMBEDDING_SCHEMA_VERSION = 1


class OnnxSearchImageEmbeddingProvider:
    """CPU inference cho projection ảnh PE-Core; input luôn là tensor chuẩn hóa 1×3×384×384."""

    def __init__(
        self,
        model_path: str,
        model_version: str,
        dimensions: int,
        image_size: int,
        intra_op_threads: int,
    ) -> None:
        # Validate sidecar/checksum trước khi tạo session để artifact không khớp bị fail-closed khi startup.
        artifact = Path(model_path)
        metadata_path = artifact.with_suffix(artifact.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schemaVersion") != IMAGE_EMBEDDING_SCHEMA_VERSION
            or metadata.get("modelVersion") != model_version
            or metadata.get("dimensions") != dimensions
            or metadata.get("imageSize") != image_size
            or metadata.get("preprocessVersion") != IMAGE_PREPROCESS_VERSION
        ):
            raise ValueError("IMAGE_EMBEDDING_METADATA_MISMATCH")
        if not artifact.is_file() or not compare_digest(self._sha256_file(artifact), str(metadata.get("artifactSha256", ""))):
            raise ValueError("IMAGE_EMBEDDING_ARTIFACT_CHECKSUM_MISMATCH")
        self.model_version = model_version
        self.dimensions = dimensions
        self.image_size = image_size
        self.mean = self._float_triplet(metadata.get("mean"), "mean")
        self.std = self._float_triplet(metadata.get("std"), "std")
        self.crop_pct = float(metadata.get("cropPct", 0))
        self.interpolation = str(metadata.get("interpolation", ""))
        if not 0 < self.crop_pct <= 1 or self.interpolation not in {"bicubic", "bilinear", "lanczos"} or min(self.std) <= 0:
            raise ValueError("IMAGE_EMBEDDING_PREPROCESS_CONFIG_INVALID")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, intra_op_threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(artifact),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        expected_shape = [1, 3, image_size, image_size]
        if (
            len(inputs) != 1
            or inputs[0].type != "tensor(float)"
            or inputs[0].shape != expected_shape
            or len(outputs) != 1
            or outputs[0].type != "tensor(float)"
            or outputs[0].shape != [1, dimensions]
        ):
            raise ValueError("IMAGE_EMBEDDING_ONNX_INPUT_SCHEMA_MISMATCH")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    # Từ chối metadata thiếu/sai chiều để preprocessor không âm thầm dùng chuẩn hóa mặc định.
    @staticmethod
    def _float_triplet(value: Any, field: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3 or any(not isinstance(item, (int, float)) for item in value):
            raise ValueError(f"IMAGE_EMBEDDING_{field.upper()}_INVALID")
        values = tuple(float(item) for item in value)
        if not all(np.isfinite(values)):
            raise ValueError(f"IMAGE_EMBEDDING_{field.upper()}_INVALID")
        return values  # type: ignore[return-value]

    # Hash theo chunk để xác nhận artifact mà không cần đọc toàn bộ model vào một Buffer.
    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # Chỉ trả raw projection; application service kiểm tra schema, độ dài và L2-normalize một lần.
    def encode(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.asarray(
            self._session.run([self._output_name], {self._input_name: tensor})[0],
            dtype=np.float32,
        )


# Artifact thiếu/sai chỉ làm image search unavailable; các router text tiếp tục hoạt động.
def build_search_image_embedding_provider(settings: Settings) -> OnnxSearchImageEmbeddingProvider | None:
    if not settings.image_search_model_path:
        logger.warning("Image embedding unavailable: model path is not configured")
        return None
    try:
        return OnnxSearchImageEmbeddingProvider(
            settings.image_search_model_path,
            settings.image_search_model_version,
            settings.image_search_dimensions,
            settings.image_search_image_size,
            settings.image_search_intra_op_threads,
        )
    except Exception as error:
        logger.warning("Image embedding unavailable: %s", error.__class__.__name__)
        return None
