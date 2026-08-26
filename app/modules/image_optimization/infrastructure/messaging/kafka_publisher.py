"""Kafka publisher cho event image optimization, chi gui metadata va khong gui binary."""

import json
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.modules.image_optimization.domain.models import ImageOptimizationJob


class KafkaOptimizationEventPublisher:
    """Adapter Kafka lazy-start de API khong fail khi local chua co broker."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._producer: Any = None

    async def start(self) -> None:
        """Khoi dong producer khi composition root bat dau process."""

        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(bootstrap_servers=self._settings.kafka_bootstrap_servers)
        await self._producer.start()

    async def stop(self) -> None:
        """Dong producer tranh task Kafka bi ro khi shutdown."""

        if self._producer is not None:
            await self._producer.stop()

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Gui event metadata; worker xu ly idempotent theo jobId khi Kafka redelivery."""

        if self._producer is None:
            raise RuntimeError("Kafka producer has not been started")
        payload = {
            "eventId": str(job.job_id),
            "eventType": "ai.image-optimization.requested",
            "schemaVersion": 1,
            "jobId": str(job.job_id),
            "productId": str(job.product_id),
            "sellerOwnerId": str(job.seller_owner_id),
            "sourceAssetIds": [str(asset_id) for asset_id in job.source_asset_ids],
            "modes": [mode.value for mode in job.requested_modes],
            "occurredAt": datetime.now(UTC).isoformat(),
        }
        await self._producer.send_and_wait(
            self._settings.kafka_image_optimization_topic,
            json.dumps(payload).encode("utf-8"),
            key=str(job.job_id).encode("utf-8"),
        )
