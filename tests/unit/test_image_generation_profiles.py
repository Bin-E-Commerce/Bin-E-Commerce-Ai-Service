"""Kiểm thử policy preview/final và ưu tiên asset đầu tiên của image optimization."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.modules.image_optimization.application.events import ImageOptimizationRequestedEvent
from app.modules.image_optimization.application.ports import GeneratedImage
from app.modules.image_optimization.application.processor import ImageOptimizationJobProcessor
from app.modules.image_optimization.domain.enums import ImageGenerationProfile, ImageOptimizationMode, ImageOptimizationStatus
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob
from app.modules.image_optimization.infrastructure.repository import InMemoryImageOptimizationJobRepository


class _Media:
    """Fake media client ghi lại source được tải và output được upload."""

    def __init__(self) -> None:
        self.downloaded: list[str] = []
        self.uploaded = 0

    # Chỉ trả source giả để kiểm thử policy, không cần đọc ảnh thật.
    async def download_source(self, *, seller_owner_id: UUID, asset_id: UUID) -> tuple[bytes, str, str]:
        """Ghi asset ID để xác nhận preview không tải toàn bộ ảnh."""

        del seller_owner_id
        self.downloaded.append(str(asset_id))
        return b"source", "image/jpeg", "source.jpg"

    # Tạo asset giả cho mỗi output provider trả về.
    async def upload_output(self, *, seller_owner_id: UUID, job_id: UUID, output: GeneratedImage) -> GeneratedAsset:
        """Đếm upload mà không lưu binary."""

        del seller_owner_id, job_id, output
        self.uploaded += 1
        return GeneratedAsset(asset_id=uuid4(), public_url="https://cdn.test/output.jpg", mode="pending")

    # Preview thành công không được gọi cleanup.
    async def cleanup_outputs(self, *, seller_owner_id: UUID, job_id: UUID) -> None:
        """Giữ contract MediaAssetClient cho processor."""

        del seller_owner_id, job_id


class _Provider:
    """Fake local provider để kiểm thử số lần sinh output."""

    # Sinh output nhỏ, không gọi provider trả phí.
    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        """Trả bytes giả cho pipeline local."""

        del source, file_name
        return GeneratedImage(content=b"output", content_type="image/jpeg", file_name="output.jpg")


# Event preview chỉ phát asset đầu tiên để giảm latency bản xem nhanh.
def test_preview_event_contains_only_first_source_asset() -> None:
    """Kiểm tra contract Kafka vẫn giữ đầy đủ asset ở profile final."""

    sources = (uuid4(), uuid4(), uuid4())
    job = ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=sources,
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="profile-event-test",
        expected_product_updated_at=datetime.now(UTC),
    )

    preview = ImageOptimizationRequestedEvent.from_job(job)
    final = ImageOptimizationRequestedEvent.from_job(
        ImageOptimizationJob.create(
            seller_owner_id=job.seller_owner_id,
            product_id=job.product_id,
            source_asset_ids=sources,
            requested_modes=job.requested_modes,
            idempotency_key="profile-final-test",
            expected_product_updated_at=job.expected_product_updated_at,
            generation_profile=ImageGenerationProfile.FINAL,
        )
    )

    assert preview.source_asset_ids == sources[:1]
    assert final.source_asset_ids == sources


# Processor preview chỉ tải và xử lý source đầu tiên của job nhiều ảnh.
@pytest.mark.asyncio
async def test_processor_preview_processes_first_source_only() -> None:
    """Đảm bảo số provider call preview không tăng theo số ảnh seller chọn."""

    repository = InMemoryImageOptimizationJobRepository()
    media = _Media()
    source_ids = (uuid4(), uuid4(), uuid4())
    job = ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=source_ids,
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="profile-processor-test",
        expected_product_updated_at=datetime.now(UTC),
    )
    await repository.save(job)

    await ImageOptimizationJobProcessor(repository, media, _Provider()).execute(job.job_id)

    stored = await repository.find_by_id(job.job_id)
    assert stored is not None
    assert stored.status is ImageOptimizationStatus.REVIEW_REQUIRED
    assert media.downloaded == [str(source_ids[0])]
    assert media.uploaded == 1


# Finalization phải được worker claim và chỉ tải source tương ứng output seller đã chọn.
@pytest.mark.asyncio
async def test_processor_finalization_claims_job_and_processes_selected_source_only() -> None:
    """AAA: finalization không bị kẹt ở FINALIZING và không chạy lại ảnh seller không chọn."""

    repository = InMemoryImageOptimizationJobRepository()
    media = _Media()
    source_ids = (uuid4(), uuid4(), uuid4())
    preview_asset = GeneratedAsset(
        asset_id=uuid4(),
        public_url="https://cdn.test/preview.jpg",
        mode="WHITE_BACKGROUND",
        source_asset_id=source_ids[2],
    )
    job = ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=source_ids,
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="profile-final-worker-test",
        expected_product_updated_at=datetime.now(UTC),
    )
    job = job.transition(ImageOptimizationStatus.PROCESSING)
    job = job.with_outputs(
        (preview_asset,),
        provider="local",
        model=None,
        prompt_version=None,
        retention_days=1,
    ).transition(ImageOptimizationStatus.REVIEW_REQUIRED)
    finalizing = job.request_finalization((preview_asset.asset_id,))
    assert finalizing.source_asset_ids == (source_ids[2],)
    await repository.save(finalizing)

    await ImageOptimizationJobProcessor(repository, media, _Provider()).execute(finalizing.job_id)

    stored = await repository.find_by_id(finalizing.job_id)
    assert stored is not None
    assert stored.status is ImageOptimizationStatus.REVIEW_REQUIRED
    assert media.downloaded == [str(source_ids[2])]


# Finalization chuyển profile sang medium và lưu lựa chọn output trước khi publish event.
def test_job_request_finalization_switches_to_final_profile() -> None:
    """Seller chọn output preview thì job phải bước vào FINALIZING."""

    source_id = uuid4()
    asset = GeneratedAsset(
        asset_id=uuid4(),
        public_url="https://cdn.test/a.jpg",
        mode="WHITE_BACKGROUND",
        source_asset_id=source_id,
    )
    job = ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=(source_id,),
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="profile-finalization-test",
        expected_product_updated_at=datetime.now(UTC),
    ).with_outputs((asset,), provider="local", model=None, prompt_version=None, retention_days=1)
    job = job.transition(ImageOptimizationStatus.PROCESSING).transition(ImageOptimizationStatus.REVIEW_REQUIRED)

    finalizing = job.request_finalization((asset.asset_id,))

    assert finalizing.status is ImageOptimizationStatus.FINALIZING
    assert finalizing.generation_profile is ImageGenerationProfile.FINAL
    assert finalizing.selected_output_asset_ids == (asset.asset_id,)
