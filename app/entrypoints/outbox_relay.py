"""Giữ tương thích đường dẫn relay cũ; triển khai chính thức nằm trong entrypoints.workers."""

from app.entrypoints.workers.outbox_relay import main, run_outbox_relay

__all__ = ["main", "run_outbox_relay"]

if __name__ == "__main__":
    main()
