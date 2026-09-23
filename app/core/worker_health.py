"""Quản lý health marker cho worker nền.

Module chỉ ghi một file heartbeat cục bộ để Kubernetes kiểm tra process đã khởi tạo
dependency nền tảng và còn tiến triển; module không tự mở socket, gọi mạng hoặc ghi log.
"""

from pathlib import Path


# Ghi heartbeat sau mỗi vòng Kafka thành công để probe không chỉ kiểm tra PID còn sống.
def mark_worker_healthy(path: str) -> None:
    """Tạo hoặc cập nhật mtime của marker, không ghi secret hay payload nghiệp vụ."""

    health_path = Path(path)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.touch(exist_ok=True)


# Xóa marker khi shutdown để lần khởi động sau phải xác nhận dependency lại từ đầu.
def clear_worker_health(path: str) -> None:
    """Xóa marker nếu tồn tại và không làm lộ lỗi phụ trong quá trình shutdown."""

    Path(path).unlink(missing_ok=True)
