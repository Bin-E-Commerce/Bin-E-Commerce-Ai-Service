"""Entrypoint huấn luyện ranking offline, tách khỏi API và các Kafka worker."""

from app.modules.ranking.infrastructure.training.lightgbm_trainer import main

if __name__ == "__main__":
    main()
