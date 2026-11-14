"""Load ONNX Cross-Encoder khi artifact và tokenizer cùng contract sẵn sàng."""

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.core.config import Settings
from app.modules.search_ranking.domain.models import SEARCH_RERANKER_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class OnnxSearchReranker:
    """Tokenizer cặp query-document và chạy inference CPU với batch nhỏ, cố định."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        version: str,
        max_sequence_length: int,
        intra_op_threads: int,
    ) -> None:
        artifact_path = Path(model_path)
        tokenizer_file = Path(tokenizer_path)
        metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schemaVersion") != SEARCH_RERANKER_SCHEMA_VERSION
            or metadata.get("modelVersion") != version
            or metadata.get("objective") != "pairwise-ranking"
            or metadata.get("maxSequenceLength") != max_sequence_length
            or metadata.get("outputDimension") != 1
        ):
            raise ValueError("SEARCH_RERANKER_METADATA_MISMATCH")
        if not tokenizer_file.is_file():
            raise FileNotFoundError("SEARCH_RERANKER_TOKENIZER_UNAVAILABLE")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, intra_op_threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(artifact_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise ValueError("SEARCH_RERANKER_INPUT_SCHEMA_MISMATCH")
        if self._input_names - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("SEARCH_RERANKER_UNSUPPORTED_INPUT")

        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=max_sequence_length)
        pad_id = next(
            (
                candidate_id
                for candidate in ("[PAD]", "<pad>", "<|pad|>")
                if (candidate_id := self._tokenizer.token_to_id(candidate)) is not None
            ),
            None,
        )
        if pad_id is None:
            raise ValueError("SEARCH_RERANKER_PADDING_TOKEN_UNAVAILABLE")
        pad_token = self._tokenizer.id_to_token(pad_id)
        if pad_token is None:
            raise ValueError("SEARCH_RERANKER_PADDING_TOKEN_INVALID")
        self._tokenizer.enable_padding(
            length=max_sequence_length,
            pad_id=pad_id,
            pad_token=pad_token,
        )
        self._batch_size = 16
        self.model_version = version

    # Chạy từng lô để bộ nhớ ổn định; model chỉ nhận chuỗi catalog đã được caller giới hạn.
    def predict(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(documents), self._batch_size):
            chunk = documents[offset : offset + self._batch_size]
            encodings = self._tokenizer.encode_batch([(query, document) for document in chunk])
            input_values: dict[str, np.ndarray[Any, Any]] = {
                "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                input_values["token_type_ids"] = np.asarray([item.type_ids for item in encodings], dtype=np.int64)
            output = self._session.run(None, input_values)[0]
            values = np.asarray(output).reshape(len(chunk), -1)
            if values.shape[1] != 1:
                raise ValueError("SEARCH_RERANKER_OUTPUT_SCHEMA_MISMATCH")
            scores.extend(float(value) for value in values[:, 0])
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("SEARCH_RERANKER_NON_FINITE_SCORE")
        return scores


# Chỉ nạp model khi file và sidecar đủ; thiếu artifact không chặn startup các module AI khác.
def build_search_reranker(settings: Settings) -> OnnxSearchReranker | None:
    """Trả None khi chưa có artifact để Product Service dùng fallback retrieval hiện hành."""

    if not settings.search_reranker_model_path or not settings.search_reranker_tokenizer_path:
        logger.warning("Search reranker unavailable: model or tokenizer path is not configured")
        return None
    try:
        return OnnxSearchReranker(
            settings.search_reranker_model_path,
            settings.search_reranker_tokenizer_path,
            settings.search_reranker_model_version,
            settings.search_reranker_max_sequence_length,
            settings.search_reranker_intra_op_threads,
        )
    except Exception as error:
        logger.warning("Search reranker unavailable: %s", error.__class__.__name__)
        return None
