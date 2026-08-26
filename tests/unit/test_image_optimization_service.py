"""Kiem thu use case create job idempotent va publish event metadata."""

from dataclasses import replace
from uuid import uuid4

import pytest

from app.core.errors import BackgroundConfigurationError
from app.modules.image_optimization.application.commands import CreateOptimizationJobsCommand
from app.modules.image_optimization.application.service import ImageOptimizationApplicationService
from app.modules.image_optimization.domain.enums import ImageOptimizationMode
from app.modules.image_optimization.infrastructure.publisher import InMemoryOptimizationEventPublisher
from app.modules.image_optimization.infrastructure.repository import InMemoryImageOptimizationJobRepository


def _command(owner_id, product_id) -> CreateOptimizationJobsCommand:
    """Tao command batch nho cho test retry."""

    return CreateOptimizationJobsCommand(
        seller_owner_id=owner_id,
        product_ids=(product_id,),
        source_asset_policy="COVER_IMAGE",
        modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="idempotency-service-test",
        expected_product_updated_at=None,
    )


@pytest.mark.asyncio
async def test_create_jobs_is_idempotent_for_batch_retry() -> None:
    """AAA: request lap lai tra job cu va khong publish Kafka event trung."""

    repository = InMemoryImageOptimizationJobRepository()
    publisher = InMemoryOptimizationEventPublisher()
    service = ImageOptimizationApplicationService(repository, publisher)
    owner_id = uuid4()
    product_id = uuid4()

    _, first_jobs = await service.create_jobs(_command(owner_id, product_id))
    _, second_jobs = await service.create_jobs(_command(owner_id, product_id))

    assert first_jobs == second_jobs
    assert len(publisher.events) == 1


# Kiểm tra request nền tùy chỉnh bị chặn an toàn khi môi trường chưa có khóa mã hóa.
@pytest.mark.asyncio
async def test_custom_background_requires_encryption_key() -> None:
    """AAA: không tạo job hoặc lưu mô tả rõ khi thiếu cấu hình bảo mật."""

    repository = InMemoryImageOptimizationJobRepository()
    publisher = InMemoryOptimizationEventPublisher()
    service = ImageOptimizationApplicationService(repository, publisher)
    command = _command(uuid4(), uuid4())
    command = replace(command, background_description="a clean studio background")

    with pytest.raises(BackgroundConfigurationError):
        await service.create_jobs(command)

    assert publisher.events == []
