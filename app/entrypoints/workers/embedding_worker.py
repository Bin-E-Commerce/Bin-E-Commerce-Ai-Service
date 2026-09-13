"""Entrypoint worker embedding; implementation Kafka nằm trong bounded context embeddings."""

from app.modules.embeddings.infrastructure.messaging.worker import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
