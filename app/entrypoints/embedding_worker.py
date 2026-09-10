"""Kafka worker tạo product embedding và phát generated event về Recommendation Service."""

import asyncio
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import ConsumerRecord, OffsetAndMetadata, TopicPartition

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.modules.embeddings.infrastructure.openai_provider import EmbeddingProviderUnavailableError, ProductEmbeddingProvider

logger = logging.getLogger(__name__)


# Validate envelope/payload trước khi gọi provider để malformed event đi DLQ mà không retry vô hạn.
def _parse(value: bytes) -> dict[str, object]:
    payload = json.loads(value.decode("utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("eventName") != "recommendation.product_embedding.requested"
        or payload.get("eventVersion") != 1
        or not isinstance(payload.get("eventId"), str)
        or not isinstance(payload.get("source"), str)
        or not isinstance(payload.get("aggregateId"), str)
        or not isinstance(payload.get("occurredAt"), str)
    ):
        raise ValueError("INVALID_EMBEDDING_REQUEST_EVENT")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("INVALID_EMBEDDING_REQUEST_DATA")
    required = ("jobId", "productId", "contentHash", "modelVersion", "text", "embeddingProfile")
    if any(not isinstance(data.get(key), str) or not str(data[key]).strip() for key in required):
        raise ValueError("INVALID_EMBEDDING_REQUEST_FIELDS")
    if data.get("embeddingProfile") != "product-content-v1":
        raise ValueError("UNSUPPORTED_EMBEDDING_PROFILE")
    text = str(data["text"])
    if len(text) > 12000:
        raise ValueError("EMBEDDING_TEXT_TOO_LONG")
    return dict(payload)


# Gửi message lỗi nguyên bản sang DLQ nhưng không ghi payload vào log vì text có thể chứa catalog private data.
async def _dlq(producer: AIOKafkaProducer, topic: str, message: ConsumerRecord[bytes, bytes]) -> None:
    await producer.send_and_wait(topic, value=message.value, key=message.key)


# Xử lý tuần tự trong từng partition để offset chỉ commit sau khi generated/DLQ đã được broker xác nhận.
async def run_worker() -> None:
    configure_logging()
    settings = get_settings()
    provider = ProductEmbeddingProvider(settings)
    consumer = AIOKafkaConsumer(
        settings.embedding_requested_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.embedding_consumer_group,
        enable_auto_commit=False,
    )
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    await consumer.start()
    await producer.start()
    try:
        async for message in consumer:
            try:
                event = _parse(message.value or b"")
                data = event["data"]
                assert isinstance(data, dict)
                requested_model = str(data["modelVersion"])
                model, vector = await provider.generate(
                    str(data["text"]),
                    requested_model,
                )
                if model != requested_model:
                    raise ValueError("EMBEDDING_MODEL_MISMATCH")
                if len(vector) != settings.embedding_dimensions:
                    raise ValueError("EMBEDDING_DIMENSION_MISMATCH")
                generated = {
                    "eventId": f"embedding-generated:{data['jobId']}:{data['contentHash']}:{model}",
                    "eventName": "recommendation.product_embedding.generated",
                    "eventVersion": 1,
                    "source": "ai-service",
                    "aggregateId": str(data["productId"]),
                    "occurredAt": datetime.now(UTC).isoformat(),
                    "data": {
                        "jobId": data["jobId"],
                        "productId": data["productId"],
                        "contentHash": data["contentHash"],
                        "model": model,
                        "modelVersion": model,
                        "dimensions": len(vector),
                        "vector": vector,
                    },
                }
                await producer.send_and_wait(
                    settings.embedding_generated_topic,
                    value=json.dumps(generated).encode(),
                    key=str(data["productId"]).encode(),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError, EmbeddingProviderUnavailableError):
                await _dlq(producer, settings.embedding_dlq_topic, message)
            except Exception:
                logger.exception("Embedding provider/message failed; keeping offset for retry")
                continue
            await consumer.commit({TopicPartition(message.topic, message.partition): OffsetAndMetadata(message.offset + 1, "")})
    finally:
        await consumer.stop()
        await producer.stop()


def main() -> None:
    """Entrypoint ổn định cho Docker process riêng của embedding worker."""

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Embedding worker stopped")


if __name__ == "__main__":
    main()
