"""Redis cache cho product-content result đã sanitize.

Cache chỉ lưu response bounded dưới JSON, không lưu prompt, URL ảnh, asset ID hoặc
user context. Cache key là SHA-256 do application command tạo.
"""

import json

from redis.asyncio import Redis

from app.modules.product_content.domain.models import (
    DescriptionBatch,
    GeneratedName,
    SafetyWarning,
    SuggestionBatch,
)


# Serialize hai loại result domain bằng schema tag nhỏ và TTL bắt buộc.
class RedisResultCache:
    """Từ chối payload cache không đúng shape thay vì dùng pickle không an toàn."""

    # Nhận client Redis được quản lý bởi application lifespan.
    def __init__(self, client: Redis, *, prefix: str = "ai:product-content:cache:") -> None:
        """Prefix tách namespace khỏi rate limiter và service khác."""

        self._client = client
        self._prefix = prefix

    # Đọc JSON và map về immutable domain batch; dữ liệu hỏng được coi như cache miss.
    async def get(self, key: str) -> SuggestionBatch | DescriptionBatch | None:
        """Không throw JSON/schema lỗi ra request vì cache không phải source of truth."""

        raw = await self._client.get(f"{self._prefix}{key}")
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            warnings = tuple(
                SafetyWarning(code=item["code"], field=item["field"], message=item["message"])
                for item in payload.get("warnings", [])
            )
            if payload.get("type") == "description":
                return DescriptionBatch(description=payload["description"], warnings=warnings)
            if payload.get("type") == "names":
                return SuggestionBatch(
                    suggestions=tuple(
                        GeneratedName(
                            title=item["title"],
                            reason=item["reason"],
                            recommended=bool(item["recommended"]),
                        )
                        for item in payload["suggestions"]
                    ),
                    warnings=warnings,
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return None

    # Ghi result đã validate với TTL hữu hạn; không có API ghi payload thô.
    async def set(self, key: str, value: SuggestionBatch | DescriptionBatch, ttl_seconds: int) -> None:
        """Serialize explicit fields để thay đổi domain không vô tình lưu dữ liệu mới."""

        warnings = [{"code": item.code, "field": item.field, "message": item.message} for item in value.warnings]
        if isinstance(value, DescriptionBatch):
            payload: dict[str, object] = {
                "type": "description",
                "description": value.description,
                "warnings": warnings,
            }
        else:
            payload = {
                "type": "names",
                "suggestions": [
                    {"title": item.title, "reason": item.reason, "recommended": item.recommended} for item in value.suggestions
                ],
                "warnings": warnings,
            }
        await self._client.set(
            f"{self._prefix}{key}",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ex=max(1, ttl_seconds),
        )
