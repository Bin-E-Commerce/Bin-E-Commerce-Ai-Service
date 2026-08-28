"""Use case worker xử lý một job tối ưu ảnh đã được claim.

Processor không biết Kafka hay HTTP schema. Nó giữ source/output mapping, không
retry provider trả phí và chỉ đánh dấu REVIEW_REQUIRED sau khi upload đủ output.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeVar
from uuid import UUID

from app.core.errors import AppError, ProviderUnavailableError
from app.modules.image_optimization.application.ports import (
    BackgroundDescriptionCipher,
    ImageOptimizationJobRepository,
    LifestyleBackgroundProviderPort,
    LifestyleBackgroundRequest,
    MediaAssetClient,
    WhiteBackgroundProviderPort,
)
from app.modules.image_optimization.application.provider_registry import ImageOptimizationProviderRegistry
from app.modules.image_optimization.domain.enums import (
    ImageGenerationProfile,
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
)
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob

ResultT = TypeVar("ResultT")


# Claim và xử lý một job theo lease để redelivery không gọi provider trùng.
class ImageOptimizationJobProcessor:
    """Điều phối source download, provider, media upload và state transition an toàn."""

    # Nhận port và policy runtime; paid provider không dùng max_retry_attempts.
    def __init__(
        self,
        repository: ImageOptimizationJobRepository,
        media_client: MediaAssetClient,
        white_provider: WhiteBackgroundProviderPort,
        lifestyle_provider: LifestyleBackgroundProviderPort | None = None,
        max_retry_attempts: int = 3,
        background_cipher: BackgroundDescriptionCipher | None = None,
        *,
        worker_id: str = "image-worker",
        lease_seconds: int = 300,
        retention_days: int = 30,
        provider_registry: ImageOptimizationProviderRegistry | None = None,
    ) -> None:
        """Lưu dependency và giới hạn retry chỉ dành cho pipeline local miễn phí."""

        self._repository = repository
        self._media_client = media_client
        self._providers = provider_registry or ImageOptimizationProviderRegistry(
            white_background=white_provider,
            lifestyle_background=lifestyle_provider,
        )
        self._max_retry_attempts = max(1, max_retry_attempts)
        self._background_cipher = background_cipher
        self._worker_id = worker_id
        self._lease_seconds = max(30, lease_seconds)
        self._retention_days = max(1, retention_days)

    # Claim atomically trước mọi network/provider call; job không claim được là redelivery no-op.
    async def execute(self, job_id: UUID) -> None:
        """Không tạo output trùng khi cùng event được consumer nhận lại."""

        processing = await self._repository.claim_for_processing(
            job_id,
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if processing is None:
            return
        if processing.attempt > self._max_retry_attempts:
            failed = processing.with_failure("MAX_ATTEMPTS_EXCEEDED")
            failed = failed.with_processing_stage(ImageOptimizationProcessingStage.FAILED)
            failed = failed.transition(ImageOptimizationStatus.FAILED).release_lease()
            await self._repository.save(failed)
            return

        outputs: list[GeneratedAsset] = []
        try:
            # Preview chỉ ưu tiên asset đầu tiên để seller nhận kết quả sớm; final xử lý toàn bộ asset đã chọn.
            source_asset_ids = (
                processing.source_asset_ids[:1]
                if processing.generation_profile is ImageGenerationProfile.PREVIEW
                else processing.source_asset_ids
            )
            sources = await asyncio.gather(
                *(self._download_source(processing, source_asset_id) for source_asset_id in source_asset_ids)
            )
            processing = processing.with_processing_stage(ImageOptimizationProcessingStage.PREPARING_IMAGE)
            await self._repository.save(processing)
            processing = processing.with_processing_stage(ImageOptimizationProcessingStage.GENERATING)
            await self._repository.save(processing)

            results = await asyncio.gather(
                *(
                    self._generate_and_upload(processing, source_asset_id, source, file_name, mode)
                    for source_asset_id, source, file_name in sources
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
            providers = {output.provider for output in outputs if output.provider}
            models = {output.model for output in outputs if output.model}
            prompt_versions = {output.prompt_version for output in outputs if output.prompt_version}
            provider_name = next(iter(providers)) if len(providers) == 1 else "mixed"
            model_name = next(iter(models)) if len(models) == 1 else ("mixed" if models else None)
            prompt_version = next(iter(prompt_versions)) if len(prompt_versions) == 1 else ("mixed" if prompt_versions else None)
            completed = processing.with_outputs(
                tuple(outputs),
                provider=provider_name,
                model=model_name,
                prompt_version=prompt_version,
                retention_days=self._retention_days,
            )
            completed = completed.with_processing_stage(ImageOptimizationProcessingStage.READY)
            completed = completed.transition(ImageOptimizationStatus.REVIEW_REQUIRED).release_lease()
            await self._repository.save(completed)
        except (AppError, OSError, TimeoutError, ValueError) as error:
            # Mọi lỗi nghiệp vụ/hạ tầng trong pipeline phải kết thúc job rõ ràng, tránh để UI polling mãi ở PROCESSING.
            # AppError giữ nguyên loại lỗi an toàn (ví dụ CONFIGURATIONERROR hoặc INVALIDPROVIDERRESPONSEERROR)
            # để API trả failureCode có ý nghĩa; lỗi Python còn lại vẫn được chuẩn hóa thành tên type.
            # Ưu tiên mã lỗi ổn định của AppError để API/UI không phụ thuộc tên class Python.
            failure_code = error.code if isinstance(error, AppError) else type(error).__name__.upper()
            if outputs:
                try:
                    await self._media_client.cleanup_outputs(
                        seller_owner_id=processing.seller_owner_id,
                        job_id=processing.job_id,
                    )
                except ProviderUnavailableError:
                    failure_code = f"{failure_code}_CLEANUP_PENDING"
            failed = processing.with_failure(failure_code).with_processing_stage(ImageOptimizationProcessingStage.FAILED)
            failed = failed.transition(ImageOptimizationStatus.FAILED).release_lease()
            await self._repository.save(failed)

    # Tải source kèm asset ID để mapping không bị mất khi xử lý song song.
    async def _download_source(self, job: ImageOptimizationJob, source_asset_id: UUID) -> tuple[UUID, bytes, str]:
        """Không trả content type vì provider nhận diện ảnh lại từ bytes thực tế."""

        source, _content_type, file_name = await self._media_client.download_source(
            seller_owner_id=job.seller_owner_id,
            asset_id=source_asset_id,
        )
        return source_asset_id, source, file_name

    # Retry có backoff chỉ cho rembg/Pillow vì không phát sinh phí theo request.
    async def _with_local_retry(self, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
        """Không được dùng helper này cho OpenAI hoặc provider trả phí."""

        for attempt in range(self._max_retry_attempts):
            try:
                return await operation()
            except ProviderUnavailableError:
                if attempt == self._max_retry_attempts - 1:
                    raise
                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
        raise ProviderUnavailableError()

    # Sinh và upload một output, đồng thời gắn source ID và metadata provider trước persistence.
    async def _generate_and_upload(
        self,
        job: ImageOptimizationJob,
        source_asset_id: UUID,
        source: bytes,
        file_name: str,
        mode: ImageOptimizationMode,
    ) -> GeneratedAsset:
        """Không retry lifestyle provider để tránh bị tính phí nhiều lần."""

        description = None
        if job.background_description_ciphertext:
            if self._background_cipher is None:
                raise ProviderUnavailableError()
            description = self._background_cipher.decrypt(job.background_description_ciphertext)
        request = LifestyleBackgroundRequest(
            preset=job.background_preset,
            description=description,
            profile=job.generation_profile,
        )

        # Chỉ capability local miễn phí được retry; provider trả phí luôn được gọi đúng một lần.
        if self._providers.is_local(mode):
            generated = await self._with_local_retry(
                lambda: self._providers.generate(
                    mode=mode,
                    source=source,
                    file_name=file_name,
                    lifestyle_request=request,
                )
            )
        else:
            generated = await self._providers.generate(
                mode=mode,
                source=source,
                file_name=file_name,
                lifestyle_request=request,
            )

        uploaded = await self._media_client.upload_output(
            seller_owner_id=job.seller_owner_id,
            job_id=job.job_id,
            output=generated,
        )
        metadata = generated.metadata
        return replace(
            uploaded,
            source_asset_id=source_asset_id,
            mode=mode.value,
            provider=metadata.provider if metadata else "unreported-provider",
            model=metadata.model if metadata else None,
            prompt_version=metadata.prompt_version if metadata else None,
        )
