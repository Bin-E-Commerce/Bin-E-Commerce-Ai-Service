"""Kiểm thử các invariant P0 của image optimization use cases.

Test dùng memory repository và fake ports; không gọi PostgreSQL, Kafka, Media,
Product Service hoặc provider AI thật.
"""

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.core.errors import IdempotencyKeyReusedError, InvalidInputError, OptimizationJobNotReadyError
from app.modules.image_optimization.application.commands import (
    ApplyOptimizationOutputsCommand,
    CreateOptimizationJobsCommand,
)
from app.modules.image_optimization.application.ports import GeneratedImage
from app.modules.image_optimization.application.processor import ImageOptimizationJobProcessor
from app.modules.image_optimization.application.service import ImageOptimizationApplicationService
from app.modules.image_optimization.application.use_cases.apply_outputs import ApplyImageOptimizationOutputs
from app.modules.image_optimization.domain.enums import ImageOptimizationMode, ImageOptimizationStatus
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob
from app.modules.image_optimization.infrastructure.publisher import InMemoryOptimizationEventPublisher
from app.modules.image_optimization.infrastructure.repository import InMemoryImageOptimizationJobRepository


# Fake Product Service ghi lại đúng output mà use case đã xác minh.
class _ProductMediaClient:
    """Không thực hiện network; chỉ lưu assets nhận được để assertion."""

    # Khởi tạo trạng thái capture cho một test.
    def __init__(self) -> None:
        """Danh sách rỗng chứng minh chưa có side effect trước execute."""

        self.applied_assets: tuple[GeneratedAsset, ...] = ()

    # Capture assets và bỏ các metadata không cần cho assertion này.
    async def apply_media(
        self,
        *,
        seller_owner_id: UUID,
        product_id: UUID,
        job_id: UUID,
        expected_product_updated_at: datetime | None,
        assets: tuple[GeneratedAsset, ...],
        permissions: tuple[str, ...] = (),
    ) -> None:
        """Mô phỏng downstream apply thành công."""

        del seller_owner_id, product_id, job_id, expected_product_updated_at, permissions
        self.applied_assets = assets

    # Method đủ protocol nhưng không được gọi trong apply tests.
    async def rollback_media(self, *, seller_owner_id: UUID, product_id: UUID, job_id: UUID) -> None:
        """Raise nếu test vô tình chạy sai workflow."""

        del seller_owner_id, product_id, job_id
        raise AssertionError("rollback_media must not be called")


# Tạo job REVIEW_REQUIRED có hai output cùng source để test selection.
def _review_job(owner_id: UUID, product_id: UUID) -> tuple[ImageOptimizationJob, tuple[GeneratedAsset, ...]]:
    """Trả aggregate hợp lệ và outputs để test không sửa trực tiếp state field."""

    source_id = uuid4()
    outputs = (
        GeneratedAsset(
            asset_id=uuid4(), public_url="https://cdn.test/one.webp", mode="WHITE_BACKGROUND", source_asset_id=source_id
        ),
        GeneratedAsset(
            asset_id=uuid4(), public_url="https://cdn.test/two.webp", mode="LIFESTYLE_BACKGROUND", source_asset_id=source_id
        ),
    )
    job = ImageOptimizationJob.create(
        seller_owner_id=owner_id,
        product_id=product_id,
        source_asset_ids=(source_id,),
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND, ImageOptimizationMode.LIFESTYLE_BACKGROUND),
        idempotency_key="apply-selection-test",
        expected_product_updated_at=datetime(2026, 8, 26, tzinfo=UTC),
    )
    job = job.transition(ImageOptimizationStatus.PROCESSING)
    job = job.with_outputs(outputs, provider="mixed", model="mixed", prompt_version="v3", retention_days=30)
    return job.transition(ImageOptimizationStatus.REVIEW_REQUIRED), outputs


# Xác nhận request images chỉ apply output seller chọn thay vì toàn bộ output job.
@pytest.mark.asyncio
async def test_apply_honors_selected_output_assets() -> None:
    """AAA: selection được đối chiếu bằng asset ID và URL lấy từ aggregate."""

    repository = InMemoryImageOptimizationJobRepository()
    owner_id = uuid4()
    job, outputs = _review_job(owner_id, uuid4())
    await repository.save(job)
    client = _ProductMediaClient()
    use_case = ApplyImageOptimizationOutputs(repository, client, allow_memory_without_downstream=False)

    updated = await use_case.execute(
        ApplyOptimizationOutputsCommand(
            job_id=job.job_id,
            seller_owner_id=owner_id,
            expected_product_updated_at=job.expected_product_updated_at,
            selected_asset_ids=(outputs[1].asset_id,),
            permissions=frozenset({"seller.ai.image_optimization.apply"}),
        )
    )

    assert updated.status is ImageOptimizationStatus.APPLIED
    assert client.applied_assets == (outputs[1],)


# Xác nhận asset không thuộc job bị chặn trước khi Product Service được gọi.
@pytest.mark.asyncio
async def test_apply_rejects_output_from_another_job() -> None:
    """AAA: browser không thể chèn asset ID tùy ý vào request apply."""

    repository = InMemoryImageOptimizationJobRepository()
    owner_id = uuid4()
    job, _outputs = _review_job(owner_id, uuid4())
    await repository.save(job)
    client = _ProductMediaClient()
    use_case = ApplyImageOptimizationOutputs(repository, client, allow_memory_without_downstream=False)

    with pytest.raises(OptimizationJobNotReadyError):
        await use_case.execute(
            ApplyOptimizationOutputsCommand(
                job_id=job.job_id,
                seller_owner_id=owner_id,
                expected_product_updated_at=job.expected_product_updated_at,
                selected_asset_ids=(uuid4(),),
                permissions=frozenset({"seller.ai.image_optimization.apply"}),
            )
        )

    assert client.applied_assets == ()


# Xac nhan request apply lap lai trong luc final worker dang chay chi tra ve job hien tai.
@pytest.mark.asyncio
async def test_apply_is_idempotent_while_finalization_is_running() -> None:
    """AAA: request lap lai khong goi Product Service them lan nua."""

    repository = InMemoryImageOptimizationJobRepository()
    owner_id = uuid4()
    job, outputs = _review_job(owner_id, uuid4())
    finalizing_job = job.request_finalization((outputs[0].asset_id,))
    await repository.save(finalizing_job)
    client = _ProductMediaClient()
    use_case = ApplyImageOptimizationOutputs(repository, client, allow_memory_without_downstream=False)

    result = await use_case.execute(
        ApplyOptimizationOutputsCommand(
            job_id=finalizing_job.job_id,
            seller_owner_id=owner_id,
            expected_product_updated_at=finalizing_job.expected_product_updated_at,
            selected_asset_ids=(),
            permissions=frozenset({"seller.ai.image_optimization.apply"}),
        )
    )

    assert result is finalizing_job
    assert result.status is ImageOptimizationStatus.FINALIZING
    assert client.applied_assets == ()


# Xác nhận cùng idempotency key không thể đại diện hai payload khác nhau.
@pytest.mark.asyncio
async def test_idempotency_key_reuse_with_different_payload_returns_conflict() -> None:
    """AAA: request thứ hai không publish thêm event và không trả nhầm batch cũ."""

    repository = InMemoryImageOptimizationJobRepository()
    publisher = InMemoryOptimizationEventPublisher()
    service = ImageOptimizationApplicationService(repository, publisher)
    owner_id = uuid4()
    command = CreateOptimizationJobsCommand(
        seller_owner_id=owner_id,
        product_ids=(uuid4(),),
        source_asset_policy="COVER_IMAGE",
        modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="same-client-key",
        expected_product_updated_at=None,
    )
    await service.create_jobs(command)

    with pytest.raises(IdempotencyKeyReusedError):
        await service.create_jobs(replace(command, product_ids=(uuid4(),)))

    assert len(publisher.events) == 1


# Xác nhận batch nhiều sản phẩm dùng identity job riêng và không vướng unique idempotency legacy.
@pytest.mark.asyncio
async def test_multi_product_batch_is_rejected() -> None:
    """AAA: client key nằm ở batch, từng job dùng batch/product identity nội bộ."""

    repository = InMemoryImageOptimizationJobRepository()
    publisher = InMemoryOptimizationEventPublisher()
    service = ImageOptimizationApplicationService(repository, publisher)
    products = (uuid4(), uuid4())

    with pytest.raises(InvalidInputError):
        await service.create_jobs(
            CreateOptimizationJobsCommand(
                seller_owner_id=uuid4(),
                product_ids=products,
                source_asset_policy="COVER_IMAGE",
                modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
                idempotency_key="multi-product-client-key",
                expected_product_updated_at=None,
            )
        )

    assert len(publisher.events) == 0


# Fake Media Service giữ source bytes và tạo asset ID output trong RAM.
class _MediaClient:
    """Đếm upload để phát hiện worker xử lý trùng do event redelivery."""

    # Khởi tạo counter và output IDs deterministic theo từng lần upload.
    def __init__(self) -> None:
        """Không lưu binary sau khi upload fake hoàn tất."""

        self.upload_count = 0

    # Trả source bytes giả; fake provider không parse ảnh nên content tối thiểu là đủ.
    async def download_source(self, *, seller_owner_id: UUID, asset_id: UUID) -> tuple[bytes, str, str]:
        """Giữ asset ID ở processor thay vì nhét vào file name/prompt."""

        del seller_owner_id, asset_id
        return b"source", "image/png", "source.png"

    # Tạo media reference mới và tăng counter.
    async def upload_output(self, *, seller_owner_id: UUID, job_id: UUID, output: GeneratedImage) -> GeneratedAsset:
        """Mô phỏng Media Service upload thành công."""

        del seller_owner_id, job_id, output
        self.upload_count += 1
        return GeneratedAsset(asset_id=uuid4(), public_url="https://cdn.test/output.webp", mode="PENDING")

    # Cleanup không được gọi trong happy path.
    async def cleanup_outputs(self, *, seller_owner_id: UUID, job_id: UUID) -> None:
        """Raise nếu processor cleanup một job thành công."""

        del seller_owner_id, job_id
        raise AssertionError("cleanup_outputs must not be called")


# Fake white provider cho biết chính xác số lần pipeline local được gọi.
class _WhiteProvider:
    """Không dùng rembg; chỉ kiểm thử worker idempotency và mapping."""

    # Khởi tạo call counter.
    def __init__(self) -> None:
        """Counter bắt đầu từ zero trước event đầu tiên."""

        self.call_count = 0

    # Trả binary fake và tăng counter cho mỗi provider invocation.
    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        """Không đọc source vì mục tiêu test không phải image processing."""

        del source, file_name
        self.call_count += 1
        return GeneratedImage(content=b"output", content_type="image/webp", file_name="white.webp")


# Xác nhận Kafka redelivery không gọi provider/upload lần thứ hai và output giữ source ID.
@pytest.mark.asyncio
async def test_processor_claim_prevents_duplicate_provider_call_and_keeps_source_mapping() -> None:
    """AAA: lần execute thứ hai no-op vì job đã REVIEW_REQUIRED."""

    repository = InMemoryImageOptimizationJobRepository()
    source_id = uuid4()
    job = ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=(source_id,),
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="worker-redelivery-test",
        expected_product_updated_at=datetime.now(UTC),
    )
    await repository.save(job)
    media = _MediaClient()
    provider = _WhiteProvider()
    processor = ImageOptimizationJobProcessor(repository, media, provider, worker_id="worker-a")

    await processor.execute(job.job_id)
    await processor.execute(job.job_id)

    stored = await repository.find_by_id(job.job_id)
    assert stored is not None
    assert stored.status is ImageOptimizationStatus.REVIEW_REQUIRED
    assert stored.generated_assets[0].source_asset_id == source_id
    assert provider.call_count == 1
    assert media.upload_count == 1
