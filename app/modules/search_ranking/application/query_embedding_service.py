"""Use case tạo vector query và cache ngắn hạn bằng khóa băm không lưu raw query."""

import hashlib
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """Vector query kèm contract dimension/version cần khớp Qdrant."""

    model_version: str
    dimensions: int
    vector: tuple[float, ...]


class QueryEmbeddingProvider(Protocol):
    """Protocol tối thiểu của provider OpenAI hoặc adapter thay thế."""

    model_version: str
    dimensions: int

    async def generate(self, query: str) -> list[float]:
        """Trả vector cùng model/dimension được khai báo trên adapter."""


# Chuẩn hóa khoảng trắng và Unicode chỉ cho cache key; provider vẫn nhận query gốc đã trim.
def _cache_digest(query: str) -> str:
    """Băm query để Redis key không làm lộ từ khóa tìm kiếm dưới dạng plaintext."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SearchQueryEmbeddingService:
    """Tái sử dụng vector query trong TTL ngắn, không cache trên mỗi user/session."""

    def __init__(
        self,
        provider: QueryEmbeddingProvider,
        redis_client: Redis | None,
        cache_ttl_seconds: int,
    ) -> None:
        self._provider = provider
        self._redis = redis_client
        self._cache_ttl_seconds = cache_ttl_seconds

    # Dùng cache nếu payload đúng schema; Redis lỗi chỉ bỏ cache, không chặn semantic retrieval.
    async def embed(self, query: str) -> QueryEmbedding:
        """Trả vector của provider hiện tại, không dùng cache thuộc modelVersion khác."""

        normalized_query = query.strip()
        cache_key = (
            f"ai:search-query-vector:{self._provider.model_version}:{self._provider.dimensions}:{_cache_digest(normalized_query)}"
        )
        cached = await self._read_cache(cache_key)
        if cached is not None:
            return cached

        vector = await self._provider.generate(normalized_query)
        result = QueryEmbedding(
            model_version=self._provider.model_version,
            dimensions=self._provider.dimensions,
            vector=tuple(vector),
        )
        await self._write_cache(cache_key, result)
        return result

    # Bỏ cache hỏng hoặc lệch dimension; không tin vector đọc từ Redis là nguồn dữ liệu chuẩn.
    async def _read_cache(self, key: str) -> QueryEmbedding | None:
        if self._redis is None or self._cache_ttl_seconds <= 0:
            return None
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return None
            vector = payload.get("vector")
            if (
                payload.get("modelVersion") != self._provider.model_version
                or payload.get("dimensions") != self._provider.dimensions
                or not isinstance(vector, list)
                or len(vector) != self._provider.dimensions
            ):
                return None
            # Chuyển đổi một lần; payload Redis sai kiểu hoặc chứa NaN được bỏ như cache miss.
            normalized_vector = tuple(float(value) for value in vector)
            if not all(math.isfinite(value) for value in normalized_vector):
                return None
            return QueryEmbedding(
                model_version=self._provider.model_version,
                dimensions=self._provider.dimensions,
                vector=normalized_vector,
            )
        except (RedisError, KeyError, TypeError, ValueError, OverflowError):
            logger.warning("Search query embedding cache read failed; continuing without cache")
            return None

    # Chỉ lưu vector dưới key băm, TTL hữu hạn; lỗi cache không đổi kết quả query.
    async def _write_cache(self, key: str, result: QueryEmbedding) -> None:
        if self._redis is None or self._cache_ttl_seconds <= 0:
            return
        try:
            await self._redis.set(
                key,
                json.dumps(
                    {
                        "modelVersion": result.model_version,
                        "dimensions": result.dimensions,
                        "vector": result.vector,
                    },
                    separators=(",", ":"),
                ),
                ex=self._cache_ttl_seconds,
            )
        except RedisError:
            logger.warning("Search query embedding cache write failed; continuing without cache")
