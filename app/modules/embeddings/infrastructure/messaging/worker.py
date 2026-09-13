"""Kafka worker cho product embedding.

File này thuộc infrastructure vì sở hữu Kafka consumer/producer và provider adapter.
Application layer chỉ nên chứa port/use case, không phụ thuộc aiokafka hoặc OpenAI.
"""

import asyncio
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import ConsumerRecord, OffsetAndMetadata, TopicPartition
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.modules.embeddings.infrastructure.openai_provider import (
    EmbeddingProviderUnavailableError,
    ProductEmbeddingProvider,
)

logger = logging.getLogger(__name__)
MAX_EMBEDDING_TEXT_CHARS = 12_000


# Validate envelope/payload trước khi gọi provider để malformed event đi DLQ, không retry vô hạn.
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
    required = (
        "jobId",
        "productId",
        "contentHash",
        "modelVersion",
        "text",
        "embeddingProfile",
    )
    if any(not isinstance(data.get(key), str) or not str(data[key]).strip() for key in required):
        raise ValueError("INVALID_EMBEDDING_REQUEST_FIELDS")
    if len(str(data["text"])) > MAX_EMBEDDING_TEXT_CHARS:
        raise ValueError("EMBEDDING_TEXT_TOO_LONG")
    if data.get("embeddingProfile") != "product-content-v1":
        raise ValueError("UNSUPPORTED_EMBEDDING_PROFILE")
    return dict(payload)


# Gửi message lỗi nguyên bản sang DLQ nhưng không ghi raw text vào log vì catalog có thể chứa dữ liệu private.
async def _send_dlq(
    producer: AIOKafkaProducer,
    topic: str,
    message: ConsumerRecord[bytes, bytes],
) -> None:
    await producer.send_and_wait(topic, value=message.value, key=message.key)


# Lưu generated payload trước khi publish để Kafka redelivery không gọi provider trả phí lần hai.
async def _get_cached_result(cache: Redis | None, key: str) -> bytes | None:
    if cache is None:
        return None
    try:
        value = await cache.get(key)
        return value if isinstance(value, bytes) else None
    except Exception:
        logger.warning("Embedding result cache unavailable; processing without shared idempotency")
        return None


# Cache chỉ chứa event generated đã bounded; không log vector hoặc raw semantic text khi Redis lỗi.
async def _cache_result(
    cache: Redis | None,
    key: str,
    value: bytes,
    ttl_seconds: int,
) -> None:
    if cache is None:
        return
    try:
        await cache.set(key, value, ex=ttl_seconds)
    except Exception:
        logger.warning("Unable to persist embedding result idempotency cache")


# Xử lý tuần tự trong từng partition và chỉ commit sau generated/DLQ được broker xác nhận.
# Xử lý tuần tự giữ ordering đơn giản; scale ngang nên dùng thêm worker instance/partition.
async def run_worker() -> None:
    configure_logging()
    settings = get_settings()
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        # Không có key thì semantic pipeline chưa bật; thoát sạch để Compose không tạo restart loop vô hạn.
        logger.warning("Embedding worker disabled: OPENAI_API_KEY is not configured")
        return

    provider = ProductEmbeddingProvider(settings)
    embedding_cache: Redis | None = None
    if settings.redis_url:
        try:
            embedding_cache = Redis.from_url(settings.redis_url, decode_responses=False)
            await embedding_cache.ping()
        except Exception:
            logger.warning("Embedding result cache unavailable; worker will continue without shared idempotency")
            if embedding_cache is not None:
                await embedding_cache.aclose()
            embedding_cache = None
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
                if not isinstance(data, dict):
                    raise ValueError("INVALID_EMBEDDING_REQUEST_DATA")

                requested_model_version = str(data["modelVersion"])
                # modelVersion là version artifact, còn embedding_model là tên model provider.
                # Worker chỉ xử lý version được cài đặt, không để message tùy ý gọi model OpenAI.
                if requested_model_version != settings.embedding_model_version:
                    raise ValueError("UNSUPPORTED_EMBEDDING_MODEL_VERSION")
                cache_key = (
                    f"ai:embedding-result:{data['jobId']}:{data['contentHash']}"
                    f":{requested_model_version}:{settings.embedding_dimensions}"
                )
                generated_bytes = await _get_cached_result(embedding_cache, cache_key)
                if generated_bytes is None:
                    model, vector = await provider.generate(str(data["text"]))
                    if len(vector) != settings.embedding_dimensions:
                        raise ValueError("EMBEDDING_DIMENSION_MISMATCH")

                    generated_bytes = json.dumps(
                        {
                            "eventId": f"embedding-generated:{data['jobId']}:{data['contentHash']}:{requested_model_version}",
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
                                "modelVersion": requested_model_version,
                                "dimensions": len(vector),
                                "vector": vector,
                            },
                        }
                    ).encode("utf-8")
                    # Đặt cache trước send để cả publish fail và commit fail đều không gọi provider lại.
                    await _cache_result(
                        embedding_cache,
                        cache_key,
                        generated_bytes,
                        max(60, min(settings.embedding_result_cache_ttl_seconds, 3600)),
                    )
                await producer.send_and_wait(
                    settings.embedding_generated_topic,
                    value=generated_bytes,
                    key=str(data["productId"]).encode("utf-8"),
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                KeyError,
                EmbeddingProviderUnavailableError,
            ):
                await _send_dlq(producer, settings.embedding_dlq_topic, message)
            except Exception:
                # Lỗi publish/provider ngoài nhóm payload lỗi phải giữ offset để Kafka redeliver.
                logger.exception("Embedding provider/message failed; keeping offset for retry")
                continue

            await consumer.commit({TopicPartition(message.topic, message.partition): OffsetAndMetadata(message.offset + 1, "")})
    finally:
        await consumer.stop()
        await producer.stop()
        if embedding_cache is not None:
            await embedding_cache.aclose()


# Khởi chạy event loop và cho phép Ctrl+C dừng worker sạch sẽ.
def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Embedding worker stopped")
