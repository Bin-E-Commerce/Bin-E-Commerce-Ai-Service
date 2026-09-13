"""Giữ tương thích đường dẫn worker cũ; triển khai chính thức nằm trong entrypoints.workers."""

from app.entrypoints.workers.image_optimization_worker import main, run_worker

__all__ = ["main", "run_worker"]

if __name__ == "__main__":
    main()
