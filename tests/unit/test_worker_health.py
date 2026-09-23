"""Kiểm tra health marker cục bộ dùng bởi probe Kubernetes của worker."""

from pathlib import Path

from app.core.worker_health import clear_worker_health, mark_worker_healthy


# Đảm bảo marker được tạo và cập nhật tại đúng đường dẫn mà probe sẽ đọc.
def test_mark_worker_healthy_creates_marker(tmp_path: Path) -> None:
    # Arrange
    marker = tmp_path / "nested" / "worker.health"

    # Act
    mark_worker_healthy(str(marker))

    # Assert
    assert marker.is_file()


# Đảm bảo shutdown có thể gọi nhiều lần mà không biến dọn dẹp thành lỗi process.
def test_clear_worker_health_is_idempotent(tmp_path: Path) -> None:
    # Arrange
    marker = tmp_path / "worker.health"
    mark_worker_healthy(str(marker))

    # Act
    clear_worker_health(str(marker))
    clear_worker_health(str(marker))

    # Assert
    assert not marker.exists()
