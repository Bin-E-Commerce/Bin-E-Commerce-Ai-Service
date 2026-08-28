"""Kiểm tra source Python, migration và test luôn là UTF-8 hợp lệ, không bị mojibake."""

from __future__ import annotations

from pathlib import Path

# Những chuỗi này thường xuất hiện khi file UTF-8 bị đọc/ghi nhầm qua Windows-1252.
MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Å", "áº", "Æ°", "�")


# Đọc từng file text trong các thư mục quality gate và trả về lỗi theo đường dẫn tương đối.
def find_encoding_issues(root: Path) -> list[str]:
    """Phát hiện byte không decode được hoặc ký tự mojibake trong source cần kiểm tra."""

    issues: list[str] = []
    for directory in ("app", "tests", "migrations"):
        directory_path = root / directory
        if not directory_path.exists():
            continue
        for path in directory_path.rglob("*.py"):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                issues.append(f"{path}: invalid UTF-8")
                continue
            if any(marker in content for marker in MOJIBAKE_MARKERS):
                issues.append(f"{path}: possible mojibake")
    return issues


# Chạy từ mọi working directory bằng cách neo root theo vị trí script.
def main() -> int:
    """In lỗi rõ ràng và trả exit code khác 0 để CI/local check dừng an toàn."""

    issues = find_encoding_issues(Path(__file__).resolve().parents[1])
    if issues:
        print("UTF-8 check failed:")
        print("\n".join(issues))
        return 1
    print("UTF-8 check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
