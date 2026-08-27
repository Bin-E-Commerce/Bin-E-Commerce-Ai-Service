"""Kiểm thử policy bảo mật prompt, event contract và HTTP status mapping."""

from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    InvalidInputError,
    ProviderUnavailableError,
    RateLimitExceededError,
    ResourceNotFoundError,
    UpstreamConflictError,
    UpstreamRequestError,
)
from app.modules.image_optimization.application.events import ImageOptimizationRequestedEvent
from app.modules.image_optimization.application.ports import LifestyleBackgroundRequest
from app.modules.image_optimization.application.prompts import build_lifestyle_prompt
from app.modules.image_optimization.domain.enums import LifestyleBackgroundPreset
from app.modules.image_optimization.infrastructure.clients.base import InternalHttpClient


# Xác nhận seller description bị cô lập và dữ liệu nhạy cảm không đi vào prompt lifestyle.
def test_lifestyle_prompt_redacts_sensitive_data_and_uses_delimiter() -> None:
    """Prompt giữ ý tưởng hợp lệ nhưng xóa URL/API key trước provider."""

    prompt = build_lifestyle_prompt(
        LifestyleBackgroundRequest(
            preset=LifestyleBackgroundPreset.WARM_HOME,
            description="Near a window https://internal.test sk-secret-value",
        )
    )

    assert "<seller-background>" in prompt
    assert "Near a window" in prompt
    assert "https://" not in prompt
    assert "sk-secret" not in prompt


# Xác nhận event encoder/decoder dùng đúng một contract versioned xuyên suốt outbox và worker.
def test_image_event_round_trip_preserves_identity() -> None:
    """Không thêm URL, permission, email hoặc prompt vào wire payload."""

    event = ImageOptimizationRequestedEvent(
        event_id=uuid4(),
        job_id=uuid4(),
        product_id=uuid4(),
        seller_owner_id=uuid4(),
        source_asset_ids=(uuid4(),),
        modes=("WHITE_BACKGROUND",),
        occurred_at=datetime.now(UTC),
    )

    parsed = ImageOptimizationRequestedEvent.from_payload(event.to_payload())

    assert parsed == event
    assert "imageUrl" not in event.to_payload()
    assert "permissions" not in event.to_payload()


# Event type/version lạ phải bị worker xem là poison message thay vì thử gọi provider.
def test_image_event_rejects_unknown_schema() -> None:
    """Chặn contract drift không tương thích."""

    with pytest.raises(ValueError):
        ImageOptimizationRequestedEvent.from_payload({"eventType": "unknown", "schemaVersion": 2})


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, UpstreamRequestError),
        (401, AuthenticationError),
        (403, AuthorizationError),
        (404, ResourceNotFoundError),
        (409, UpstreamConflictError),
        (422, InvalidInputError),
        (429, RateLimitExceededError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
    ],
)
# Mỗi upstream status phải giữ đúng semantics thay vì bị gom thành 503.
def test_internal_http_client_maps_status_semantics(status_code: int, error_type: type[Exception]) -> None:
    """Không đọc raw body khi map lỗi."""

    response = httpx.Response(status_code, headers={"retry-after": "7"})

    with pytest.raises(error_type):
        InternalHttpClient.ensure_success(response)


# Response 2xx không được raise để adapter tiếp tục validate body của chính nó.
def test_internal_http_client_accepts_success_status() -> None:
    """Bao phủ nhánh success của mapper dùng chung."""

    InternalHttpClient.ensure_success(httpx.Response(204))
