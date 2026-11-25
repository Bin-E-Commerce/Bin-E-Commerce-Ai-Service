"""Export model PE-Core S/16 sang ONNX và ghi sidecar chứa preprocessing contract."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REPOSITORY = "timm/vit_pe_core_small_patch16_384.fb"
MODEL_REVISION = "255fa76e26fdf8daa47e0d607da46b7051879045"


# Hash file theo chunk để export không cần nạp thêm toàn bộ ONNX vào RAM.
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Nạp model theo revision pin; API runtime không tải checkpoint từ internet.
# Tắt fused attention khi export để ONNX không phụ thuộc SDPA/CUDA trong lúc trace.
def export_model(output_path: Path, model_version: str, revision: str) -> None:
    import timm
    import torch
    from timm.data import resolve_model_data_config
    from timm.layers import set_fused_attn

    model_id = f"hf-hub:{MODEL_REPOSITORY}@{revision}"
    # Fused SDPA đưa is_causal dạng Tensor vào trace TorchScript.
    # Tắt fused attention để ONNX export attention bằng phép toán thông thường.
    set_fused_attn(False)
    model = timm.create_model(model_id, pretrained=True)
    model.eval()
    data_config = resolve_model_data_config(model)
    input_size = data_config["input_size"]
    if input_size[0] != 3 or input_size[1] != input_size[2]:
        raise ValueError("PE_CORE_INPUT_SHAPE_UNSUPPORTED")
    image_size = int(input_size[1])
    mean = [float(value) for value in data_config["mean"]]
    std = [float(value) for value in data_config["std"]]
    crop_pct = float(data_config.get("crop_pct", 1.0))
    interpolation = str(data_config.get("interpolation", "bicubic")).lower().split(".")[-1]
    if image_size != 384 or interpolation not in {"bicubic", "bilinear", "lanczos"}:
        raise ValueError("PE_CORE_PREPROCESS_CONFIG_UNSUPPORTED")

    sample = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    with torch.inference_mode():
        output = model(sample)
    if output.ndim != 2 or output.shape != (1, 512):
        raise ValueError("PE_CORE_OUTPUT_DIMENSION_MISMATCH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(output_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
    )
    digest = _sha256_file(output_path)
    metadata = {
        "schemaVersion": 1,
        "modelVersion": model_version,
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": revision,
        "artifactSha256": digest,
        "dimensions": 512,
        "imageSize": image_size,
        "preprocessVersion": "pe-core-image-transform-v1",
        "mean": mean,
        "std": std,
        "cropPct": crop_pct,
        "interpolation": interpolation,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


# Parse đường dẫn/version rõ ràng để operator có thể tái tạo artifact giống nhau và kiểm tra checksum.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-version", default="pe-core-s16-384-r255fa76e")
    parser.add_argument("--revision", default=MODEL_REVISION)
    arguments = parser.parse_args()
    export_model(arguments.output, arguments.model_version, arguments.revision)


if __name__ == "__main__":
    main()
