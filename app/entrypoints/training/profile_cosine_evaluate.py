"""CLI chạy đối chiếu offline profileAffinity với baseline cosine trên dataset synthetic.

Entrypoint chỉ điều phối tham số và ghi report riêng; nó không train model,
không sửa dataset nguồn, không gọi database hoặc dịch vụ bên ngoài.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.modules.ranking.infrastructure.training.profile_cosine_evaluator import (
    evaluate_synthetic_profile_cosine,
)


# Parse dataset/output paths để report không ghi đè file dữ liệu hoặc metrics gốc của trainer.
def main(argv: Sequence[str] | None = None) -> None:
    """Đọc dataset synthetic, tính hai baseline và ghi report JSON riêng."""

    parser = argparse.ArgumentParser(description="Compare profileAffinity with a content-profile cosine baseline")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_synthetic_profile_cosine(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
