"""Use case worker xu ly image optimization tu event metadata.

Processor khong biet Kafka; no nhan mot job ID, tai source qua Media port, chon provider
va luu asset tham chieu. Cach tach nay giup test retry/idempotency khong can broker that.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from typing import TypeVar
from uuid import UUID

from app.core.errors import ProviderUnavailableError
from app.modules.image_optimization.domain.enums import (
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
)
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob
from app.modules.image_optimization.domain.ports import (
    BackgroundDescriptionCipher,
    ImageOptimizationJobRepository,
    LifestyleBackgroundProviderPort,
    LifestyleBackgroundRequest,
    MediaAssetClient,
    WhiteBackgroundProviderPort,
)

ResultT = TypeVar("ResultT")


# Orchestrator giữ state transition, retry và cleanup; không biết Kafka hoặc chi tiết HTTP.
class ImageOptimizationJobProcessor:
    """Orchestrate mot job idempotent va khong ghi binary/prompt vao database."""

    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        media_client: MediaAssetClient,
        white_provider: WhiteBackgroundProviderPort,
        lifestyle_provider: LifestyleBackgroundProviderPort | None = None,
        max_retry_attempts: int = 3,
        background_cipher: BackgroundDescriptionCipher | None = None,
    ) -> None:
        self._repository = repository
        self._media_client = media_client
        self._white_provider = white_provider
        self._lifestyle_provider = lifestyle_provider
        self._max_retry_attempts = max(1, max_retry_attempts)
        self._background_cipher = background_cipher

    # Đọc job, chuyển PROCESSING, xử lý các mode song song và chỉ REVIEW_REQUIRED sau khi upload đủ output.
    async def execute(self, job_id: UUID) -> None:
        """Xu ly PENDING/FAILED job; redelivery terminal se return ma khong tao output trung."""

        job = await self._repository.find_by_id(job_id)
        if job is None or job.status in {
            ImageOptimizationStatus.REVIEW_REQUIRED,
            ImageOptimizationStatus.SUCCEEDED,
            ImageOptimizationStatus.APPLIED,
            ImageOptimizationStatus.REJECTED,
            ImageOptimizationStatus.ROLLED_BACK,
        }:
            return
        if job.status is ImageOptimizationStatus.FAILED:
            job = job.transition(ImageOptimizationStatus.PENDING)
        processing = job.transition(ImageOptimizationStatus.PROCESSING).with_processing_stage(
            ImageOptimizationProcessingStage.FETCHING_SOURCE
        )
        await self._repository.save(processing)

        outputs: list[GeneratedAsset] = []
        try:
            # Tải song song các ảnh seller đã chọn; Media Service vẫn kiểm tra ownership cho từng asset.
            sources = await asyncio.gather(
                *(
                    self._media_client.download_source(
                        seller_owner_id=processing.seller_owner_id,
                        asset_id=asset_id,
                    )
                    for asset_id in processing.source_asset_ids
                )
            )
            processing = processing.with_processing_stage(ImageOptimizationProcessingStage.PREPARING_IMAGE)
            await self._repository.save(processing)
            # Hai mode dùng chung source bytes nhưng chạy song song để tổng thời gian gần bằng mode chậm nhất.
            processing = processing.with_processing_stage(ImageOptimizationProcessingStage.GENERATING)
            await self._repository.save(processing)
            results = await asyncio.gather(
                *(
                    self._generate_and_upload(processing, source, file_name, mode)
                    for source, _content_type, file_name in sources
                    for mode in processing.requested_modes
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
                outputs.append(result)

            processing = processing.with_processing_stage(ImageOptimizationProcessingStage.UPLOADING)
            await self._repository.save(processing)
            provider_name = "mixed" if len(processing.requested_modes) > 1 else self._provider_name(processing.requested_modes[0])
            model = "mixed" if len(processing.requested_modes) > 1 else self._model_name(processing.requested_modes[0])

            output_ids = tuple(output.asset_id for output in outputs)
            completed = processing.with_outputs(output_ids, provider_name, model, "image-optimization-v2", tuple(outputs))
            completed = completed.with_processing_stage(ImageOptimizationProcessingStage.READY)
            await self._repository.save(completed.transition(ImageOptimizationStatus.REVIEW_REQUIRED))
        except (ProviderUnavailableError, OSError, TimeoutError) as error:
            # Chi luu failure code on dinh; chi tiet provider khong duoc log de tranh lo du lieu nhay cam.
            if outputs:
                with suppress(Exception):
                    await self._media_client.cleanup_outputs(seller_owner_id=processing.seller_owner_id, job_id=processing.job_id)
            failed = (
                processing.with_failure(type(error).__name__.upper())
                .with_processing_stage(ImageOptimizationProcessingStage.FAILED)
                .transition(ImageOptimizationStatus.FAILED)
            )
            await self._repository.save(failed)

    async def _with_retry(self, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
        """Retry loi tam thoi theo gioi han chi phi; khong retry loi validate/business."""

        for attempt in range(self._max_retry_attempts):
            try:
                return await operation()
            except ProviderUnavailableError:
                if attempt == self._max_retry_attempts - 1:
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))
        raise ProviderUnavailableError()

    # Sinh và upload một mode độc lập; tách hàm giúp asyncio.gather giữ lỗi từng mode để cleanup đúng job.
    async def _generate_and_upload(
        self,
        job: ImageOptimizationJob,
        source: bytes,
        file_name: str,
        mode: ImageOptimizationMode,
    ) -> GeneratedAsset:
        if mode is ImageOptimizationMode.WHITE_BACKGROUND:
            generated = await self._with_retry(lambda: self._white_provider.generate_white_background(source, file_name))
        elif mode is ImageOptimizationMode.LIFESTYLE_BACKGROUND:
            provider = self._lifestyle_provider
            if provider is None:
                raise ProviderUnavailableError()
            # Không retry GPT-Image-2 tự động vì mỗi lần gọi provider có thể phát sinh chi phí; seller sẽ chủ động tạo lại.
            description = None
            if job.background_description_ciphertext:
                if self._background_cipher is None:
                    raise ProviderUnavailableError()
                # Chỉ giải mã trong RAM ngay trước provider để broker và database không biết nội dung seller viết.
                description = self._background_cipher.decrypt(job.background_description_ciphertext)
            generated = await provider.generate_lifestyle_background(
                source,
                file_name,
                LifestyleBackgroundRequest(preset=job.background_preset, description=description),
            )
        else:
            raise ProviderUnavailableError()
        uploaded = await self._media_client.upload_output(
            seller_owner_id=job.seller_owner_id,
            job_id=job.job_id,
            output=generated,
        )
        return replace(uploaded, mode=mode.value)

    # Ghi metadata provider ngắn gọn để preview/analytics biết output đến từ pipeline nào.
    @staticmethod
    def _provider_name(mode: ImageOptimizationMode) -> str:
        return "white-background-local" if mode is ImageOptimizationMode.WHITE_BACKGROUND else "openai"

    # Ghi model tương ứng mà không lưu prompt hoặc binary vào database.
    @staticmethod
    def _model_name(mode: ImageOptimizationMode) -> str:
        return "rembg-pillow" if mode is ImageOptimizationMode.WHITE_BACKGROUND else "gpt-image-2"
