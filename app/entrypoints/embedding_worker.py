"""Compatibility entrypoint cho embedding worker sau khi implementation chuyển về bounded context embeddings."""

from app.entrypoints.workers.embedding_worker import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
