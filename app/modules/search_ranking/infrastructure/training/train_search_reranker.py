"""Fine-tune multilingual Cross-Encoder từ query-product judgments rồi export ONNX.

Input là JSONL đã gán nhãn thủ công/được đánh giá có nguồn; script không tự sinh
nhãn, không đọc production database và không ghi query vào report.
"""

import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEARCH_RERANKER_SCHEMA_VERSION = "search-reranker-v1"
DEFAULT_BASE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_QUERY_CHARS = 120
MAX_PRODUCT_CHARS = 2_400
MAX_CANDIDATES_PER_QUERY = 300


@dataclass(frozen=True, slots=True)
class Candidate:
    """Candidate product với nhãn relevance 0-3 và thứ tự lexical ban đầu."""

    item_id: str
    text: str
    relevance: int
    retrieval_rank: int


@dataclass(frozen=True, slots=True)
class QueryGroup:
    """Một request tìm kiếm và tập candidate đã được đánh giá cùng query."""

    query: str
    query_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class PairExample:
    """Cặp cùng query trong đó left có nhãn relevance cao hơn right."""

    query: str
    left_text: str
    right_text: str


# Đọc JSONL và từ chối nhóm sai shape để nhãn bẩn không đi vào loss hoặc metric.
def read_groups(path: Path) -> list[QueryGroup]:
    """Parse một query mỗi dòng; candidate ID và thứ tự retrieval phải duy nhất."""

    groups: list[QueryGroup] = []
    seen_query_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                payload = json.loads(raw_line)
                query = str(payload["query"]).strip()
                query_id = str(payload["queryId"]).strip()
                raw_candidates = payload["candidates"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"INVALID_JSONL_ROW:{path}:{line_number}") from error
            if (
                not query
                or len(query) > MAX_QUERY_CHARS
                or not query_id
                or query_id in seen_query_ids
                or not isinstance(raw_candidates, list)
                or not 2 <= len(raw_candidates) <= MAX_CANDIDATES_PER_QUERY
            ):
                raise ValueError(f"INVALID_QUERY_GROUP:{path}:{line_number}")

            seen_query_ids.add(query_id)
            candidate_ids: set[str] = set()
            retrieval_ranks: set[int] = set()
            candidates: list[Candidate] = []
            for raw_candidate in raw_candidates:
                try:
                    item_id = str(raw_candidate["itemId"]).strip()
                    text = str(raw_candidate["text"]).strip()
                    raw_relevance = raw_candidate["relevance"]
                    raw_retrieval_rank = raw_candidate["retrievalRank"]
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"INVALID_CANDIDATE:{path}:{line_number}") from error
                if (
                    isinstance(raw_relevance, bool)
                    or not isinstance(raw_relevance, int)
                    or isinstance(raw_retrieval_rank, bool)
                    or not isinstance(raw_retrieval_rank, int)
                ):
                    raise ValueError(f"INVALID_CANDIDATE:{path}:{line_number}")
                relevance = raw_relevance
                retrieval_rank = raw_retrieval_rank
                if (
                    not item_id
                    or len(item_id) > 128
                    or item_id in candidate_ids
                    or not text
                    or len(text) > MAX_PRODUCT_CHARS
                    or relevance not in (0, 1, 2, 3)
                    or retrieval_rank < 1
                    or retrieval_rank in retrieval_ranks
                ):
                    raise ValueError(f"INVALID_CANDIDATE:{path}:{line_number}")
                candidate_ids.add(item_id)
                retrieval_ranks.add(retrieval_rank)
                candidates.append(Candidate(item_id, text, relevance, retrieval_rank))
            groups.append(QueryGroup(query, query_id, tuple(candidates)))

    if not groups:
        raise ValueError(f"EMPTY_DATASET:{path}")
    return groups


# Chặn cùng query text lọt qua các split để model không được chọn dựa trên câu hỏi đã thấy.
def assert_disjoint_queries(*splits: list[QueryGroup]) -> None:
    """Từ chối train/validation/test có query trùng nhau sau chuẩn hóa Unicode/case."""

    normalized_splits = [
        {
            re.sub(
                r"\s+",
                " ",
                unicodedata.normalize("NFKC", group.query).casefold().strip(),
            )
            for group in split
        }
        for split in splits
    ]
    for left_index, left in enumerate(normalized_splits):
        for right in normalized_splits[left_index + 1 :]:
            if left & right:
                raise ValueError("QUERY_TEXT_LEAKAGE_BETWEEN_SPLITS")


# Sinh các cặp preference có nhãn khác nhau; cap theo từng query ngăn query dài áp đảo epoch.
def build_pair_examples(groups: list[QueryGroup], seed: int, max_pairs: int) -> list[PairExample]:
    """Dùng label lớn hơn làm positive trong pairwise logistic ranking loss."""

    examples: list[PairExample] = []
    for group_index, group in enumerate(groups):
        rng = random.Random(seed + group_index)
        pairs: dict[tuple[str, str], PairExample] = {}
        # Lấy mẫu trực tiếp positive/negative, không tạo tích Descartes toàn bộ khi query có nhiều candidate.
        higher = [candidate for candidate in group.candidates if candidate.relevance > 0]
        lower = [candidate for candidate in group.candidates if candidate.relevance < 3]
        max_attempts = max_pairs * 20
        while len(pairs) < max_pairs and max_attempts > 0 and higher and lower:
            max_attempts -= 1
            left = rng.choice(higher)
            right = rng.choice(lower)
            if left.relevance <= right.relevance:
                continue
            pairs[(left.item_id, right.item_id)] = PairExample(group.query, left.text, right.text)
        examples.extend(pairs.values())
    if not examples:
        raise ValueError("TRAIN_DATASET_HAS_NO_RELEVANCE_PAIRS")
    return examples


# Tính NDCG graded gain, Recall positive và reciprocal rank theo từng query rồi macro-average.
def ranking_metrics(groups: list[QueryGroup], predictions: list[list[float]], k: int = 10) -> dict[str, float]:
    """Trả metric top-k cùng denominator rõ ràng cho tập query có nhãn."""

    if len(groups) != len(predictions) or k < 1:
        raise ValueError("METRIC_INPUT_INVALID")
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    for group, scores in zip(groups, predictions, strict=True):
        if len(scores) != len(group.candidates) or not all(math.isfinite(value) for value in scores):
            raise ValueError("METRIC_SCORE_INVALID")
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], group.candidates[index].retrieval_rank))
        ideal = sorted((candidate.relevance for candidate in group.candidates), reverse=True)
        dcg = sum((2 ** group.candidates[index].relevance - 1) / math.log2(rank + 2) for rank, index in enumerate(order[:k]))
        idcg = sum((2**relevance - 1) / math.log2(rank + 2) for rank, relevance in enumerate(ideal[:k]))
        ndcg_values.append(dcg / idcg if idcg else 0.0)

        relevant_count = sum(candidate.relevance > 0 for candidate in group.candidates)
        if relevant_count:
            retrieved = sum(group.candidates[index].relevance > 0 for index in order[:k])
            recall_values.append(retrieved / relevant_count)
            first_relevant = next(
                (rank + 1 for rank, index in enumerate(order) if group.candidates[index].relevance > 0),
                None,
            )
            reciprocal_ranks.append(1 / first_relevant if first_relevant else 0.0)

    return {
        "queryCount": float(len(groups)),
        "ndcgAt10": sum(ndcg_values) / len(ndcg_values),
        "recallAt10": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "queriesWithRelevantItems": float(len(recall_values)),
    }


# Giữ thứ tự pool retrieval trước Cross-Encoder để baseline dùng đúng candidate set đã được gán nhãn.
def retrieval_baseline_predictions(groups: list[QueryGroup]) -> list[list[float]]:
    """Điểm âm retrievalRank giữ nguyên thứ tự ứng viên trước bước rerank."""

    return [[-float(candidate.retrieval_rank) for candidate in group.candidates] for group in groups]


# Chạy các nhóm bằng tokenizer/model hiện tại; inference theo batch để phù hợp máy GPU phổ thông.
def predict_groups(
    model: Any, tokenizer: Any, groups: list[QueryGroup], device: Any, max_length: int, batch_size: int
) -> list[list[float]]:
    """Trả score theo đúng thứ tự candidate gốc để ghép lại relevance labels."""

    import torch

    model.eval()
    results: list[list[float]] = []
    with torch.no_grad():
        for group in groups:
            query_scores: list[float] = []
            for start in range(0, len(group.candidates), batch_size):
                candidates = group.candidates[start : start + batch_size]
                encoded = tokenizer(
                    [group.query] * len(candidates),
                    [candidate.text for candidate in candidates],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                encoded = {
                    name: value.to(device)
                    for name, value in encoded.items()
                    if name in {"input_ids", "attention_mask", "token_type_ids"}
                }
                logits = model(**encoded).logits.reshape(-1).float().cpu().tolist()
                query_scores.extend(float(score) for score in logits)
            results.append(query_scores)
    return results


# Đo artifact ONNX cuối cùng theo cùng qrels; metric deploy không được lấy từ model float trước lượng tử hóa.
def predict_onnx_groups(model: Any, groups: list[QueryGroup]) -> list[list[float]]:
    """Chạy runtime adapter với tokenizer/model đã export và trả score theo candidate gốc."""

    return [model.predict(group.query, [candidate.text for candidate in group.candidates]) for group in groups]


# Huấn luyện pairwise Cross-Encoder rồi export artifact + metadata kiểm tra được ở serving.
def train(args: argparse.Namespace) -> dict[str, object]:
    """Fine-tune model base, đánh giá test đúng một lần cuối và ghi artifact ONNX."""

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_groups = read_groups(args.train)
    validation_groups = read_groups(args.validation)
    test_groups = read_groups(args.test)
    assert_disjoint_queries(train_groups, validation_groups, test_groups)
    pairs = build_pair_examples(train_groups, args.seed, args.max_pairs_per_query)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    model.to(device)
    if device.type == "cuda":
        # Checkpointing giảm activation memory để fine-tune model 118M trên GPU 4 GB, đổi lại mỗi epoch chậm hơn.
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = device.type == "cuda" and args.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Giữ checkpoint tốt nhất trên validation, bắt đầu từ base để fine-tune không buộc làm giảm ranking quality.
    initial_validation_scores = predict_groups(model, tokenizer, validation_groups, device, args.max_length, args.eval_batch_size)
    best_validation = ranking_metrics(validation_groups, initial_validation_scores)
    best_validation_key = (
        best_validation["ndcgAt10"],
        best_validation["mrr"],
        best_validation["recallAt10"],
    )
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(pairs)
        model.train()
        total_loss = 0.0
        update_count = 0
        # Mỗi batch chứa hai document cho mỗi preference pair; padding cố định làm export/runtime giống nhau.
        for start in range(0, len(pairs), args.pair_batch_size):
            batch = pairs[start : start + args.pair_batch_size]
            texts = [pair.left_text for pair in batch] + [pair.right_text for pair in batch]
            queries = [pair.query for pair in batch] * 2
            encoded = tokenizer(
                queries,
                texts,
                max_length=args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {
                name: value.to(device)
                for name, value in encoded.items()
                if name in {"input_ids", "attention_mask", "token_type_ids"}
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                scores = model(**encoded).logits.reshape(-1)
                count = len(batch)
                loss = torch.nn.functional.softplus(-(scores[:count] - scores[count:])).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            update_count += 1

        validation_scores = predict_groups(model, tokenizer, validation_groups, device, args.max_length, args.eval_batch_size)
        validation_metrics = ranking_metrics(validation_groups, validation_scores)
        validation_key = (
            validation_metrics["ndcgAt10"],
            validation_metrics["mrr"],
            validation_metrics["recallAt10"],
        )
        is_best_checkpoint = validation_key > best_validation_key
        if is_best_checkpoint:
            # Primary metric NDCG@10 chooses checkpoint; MRR then Recall@10 break exact ties without reading test.
            best_validation = validation_metrics
            best_validation_key = validation_key
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "pairwiseLoss": total_loss / max(update_count, 1),
                    "validation": validation_metrics,
                    "selectedCheckpoint": is_best_checkpoint,
                },
                ensure_ascii=False,
            )
        )

    # Khôi phục checkpoint được chọn trước khi export; test chỉ dùng một lần để báo cáo cuối.
    model.load_state_dict(best_state)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model_dir = output_dir / "hf-model"
    model.to("cpu").eval()
    model.save_pretrained(hf_model_dir)
    tokenizer.save_pretrained(hf_model_dir)

    # Export hai input runtime cố định; tokenizer artifact được lưu cạnh model để serving không tải mạng.
    sample = tokenizer(
        "điện thoại pin lâu",
        "điện thoại thông minh pin 5000mAh",
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    dummy_inputs = [sample["input_ids"], sample["attention_mask"]]
    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in sample:
        dummy_inputs.append(sample["token_type_ids"])
        input_names.append("token_type_ids")

    # Torch chỉ là optional training dependency nên mypy không có type stub tại serving install.
    class LogitOutput(torch.nn.Module):  # type: ignore[misc]
        """Giữ graph chỉ có input ids, attention mask và một relevance logit."""

        def __init__(self, wrapped_model: Any) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        # Wrapper loại token-type input không cần thiết để runtime có một contract ổn định.
        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any | None = None,
        ) -> Any:
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            return self.wrapped_model(**model_inputs).logits

    fp32_model_path = output_dir / "search-reranker.fp32.onnx"
    model_path = output_dir / "search-reranker.onnx"
    torch.onnx.export(
        LogitOutput(model),
        tuple(dummy_inputs),
        str(fp32_model_path),
        input_names=input_names,
        output_names=["relevance_score"],
        dynamic_axes={
            **{name: {0: "batch"} for name in input_names},
            "relevance_score": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    # Dynamic int8 giảm footprint khi serving CPU; test cuối phải chạy chính artifact đã lượng tử hóa.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_model_path),
        model_output=str(model_path),
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
    fp32_model_path.unlink()
    metadata = {
        "schemaVersion": SEARCH_RERANKER_SCHEMA_VERSION,
        "modelVersion": args.model_version,
        "objective": "pairwise-ranking",
        "outputDimension": 1,
        "quantization": "dynamic-int8-per-channel",
        "maxSequenceLength": args.max_length,
        "baseModel": args.base_model,
        "trainingQueryGroups": len(train_groups),
        "validationQueryGroups": len(validation_groups),
        "testQueryGroups": len(test_groups),
        "selectedEpoch": best_epoch,
        "selectedValidation": best_validation,
        "labelScale": "0=irrelevant, 1=weakly relevant, 2=relevant, 3=highly relevant",
    }
    model_path.with_suffix(model_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    from app.modules.search_ranking.infrastructure.registry import OnnxSearchReranker

    serving_model = OnnxSearchReranker(
        str(model_path),
        str(hf_model_dir / "tokenizer.json"),
        args.model_version,
        args.max_length,
        1,
    )
    test_scores = predict_onnx_groups(serving_model, test_groups)
    baseline_metrics = ranking_metrics(test_groups, retrieval_baseline_predictions(test_groups))
    model_metrics = ranking_metrics(test_groups, test_scores)
    report = {
        "modelVersion": args.model_version,
        "baseModel": args.base_model,
        "servingArtifact": "dynamic-int8-per-channel ONNX",
        "trainingQueryGroups": len(train_groups),
        "validationQueryGroups": len(validation_groups),
        "testQueryGroups": len(test_groups),
        "selectedEpoch": best_epoch,
        "selectedValidation": best_validation,
        "test": {"preRerankCandidateOrder": baseline_metrics, "trainedCrossEncoder": model_metrics},
        "delta": {key: model_metrics[key] - baseline_metrics[key] for key in ("ndcgAt10", "recallAt10", "mrr")},
        "note": "Chỉ có ý nghĩa trên qrels test đã gán nhãn; không đại diện cho toàn bộ traffic nếu mẫu không đại diện.",
    }
    (output_dir / "evaluation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


# Parse đường dẫn split độc lập để người vận hành không vô tình dùng test làm validation.
def parse_args() -> argparse.Namespace:
    """Expose cấu hình huấn luyện có giới hạn hợp lý cho máy cá nhân."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-version", default="search-reranker-v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-pairs-per-query", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.max_length < 64
        or args.max_length > 512
        or args.pair_batch_size < 1
        or args.eval_batch_size < 1
        or args.max_pairs_per_query < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("training limits must be positive and max-length must be between 64 and 512")
    args.model_version = args.model_version.strip()
    if not args.model_version or len(args.model_version) > 80:
        parser.error("model-version must contain 1 to 80 characters")
    return args


# CLI entrypoint keeps optional torch/transformers imports out of normal API startup.
def main() -> None:
    """Train from the three explicit splits and print one machine-readable test report."""

    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
