"""SQLAlchemy repository cho batch, job và output tối ưu ảnh.

Adapter dùng exact idempotency lookup, SQL COUNT và row lock khi claim. Domain
không biết ORM; legacy JSON chỉ được đọc để tương thích dữ liệu cũ.
"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.image_optimization.domain.enums import (
    ImageGenerationProfile,
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
    LifestyleBackgroundPreset,
)
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationBatch, ImageOptimizationJob
from app.modules.image_optimization.infrastructure.persistence.models import (
    ImageOptimizationBatchRecord,
    ImageOptimizationJobRecord,
    ImageOptimizationOutputRecord,
)


# Triển khai repository production bằng async SQLAlchemy session do composition root quản lý.
class SqlAlchemyImageOptimizationJobRepository:
    """Map ORM record sang aggregate và giữ transaction boundary rõ ràng."""

    # Nhận session theo request/message, không tự tạo engine hoặc đọc settings.
    def __init__(self, session: AsyncSession) -> None:
        """Lưu session để job và outbox có thể cùng transaction."""

        self._session = session

    # Ghi batch và để unique constraint bảo vệ concurrent idempotency request.
    async def save_batch(self, batch: ImageOptimizationBatch) -> ImageOptimizationBatch:
        """Dùng ON CONFLICT để concurrent retry không biến idempotency race thành HTTP 500."""

        statement = (
            postgresql_insert(ImageOptimizationBatchRecord)
            .values(
                batch_id=batch.batch_id,
                seller_owner_id=batch.seller_owner_id,
                idempotency_key=batch.idempotency_key,
                request_hash=batch.request_hash,
                created_at=batch.created_at,
            )
            .on_conflict_do_nothing(constraint="uq_ai_image_batch_owner_key")
            .returning(ImageOptimizationBatchRecord.batch_id)
        )
        inserted_id = (await self._session.execute(statement)).scalar_one_or_none()
        if inserted_id is not None:
            return batch
        existing = await self.find_batch(batch.seller_owner_id, batch.idempotency_key)
        if existing is None:
            raise RuntimeError("Persisted idempotency batch could not be loaded")
        return existing

    # Tìm batch bằng equality trên owner/key, không dùng LIKE hoặc prefix.
    async def find_batch(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationBatch | None:
        """Trả domain batch để use case so sánh request hash."""

        statement = select(ImageOptimizationBatchRecord).where(
            ImageOptimizationBatchRecord.seller_owner_id == seller_owner_id,
            ImageOptimizationBatchRecord.idempotency_key == idempotency_key,
        )
        record = (await self._session.execute(statement)).scalar_one_or_none()
        return self._batch_to_domain(record) if record else None

    # Ghi job và output table; legacy JSON chỉ được mirror để response cũ vẫn đọc được trong một release.
    async def save(self, job: ImageOptimizationJob) -> None:
        """Upsert aggregate metadata và thay output rows theo snapshot domain hiện tại."""

        record = await self._session.get(ImageOptimizationJobRecord, job.job_id)
        if record is None:
            record = ImageOptimizationJobRecord(
                job_id=job.job_id,
                batch_id=job.batch_id,
                seller_owner_id=job.seller_owner_id,
                product_id=job.product_id,
                idempotency_key=job.idempotency_key,
                created_at=job.created_at,
            )
            self._session.add(record)
        self._copy_job(record, job)
        await self._session.flush()

        if job.generated_assets:
            await self._session.execute(
                delete(ImageOptimizationOutputRecord).where(ImageOptimizationOutputRecord.job_id == job.job_id)
            )
            self._session.add_all(
                [
                    ImageOptimizationOutputRecord(
                        output_id=asset.output_id,
                        job_id=job.job_id,
                        source_asset_id=asset.source_asset_id,
                        asset_id=asset.asset_id,
                        mode=asset.mode,
                        provider=asset.provider,
                        model=asset.model,
                        prompt_version=asset.prompt_version,
                        created_at=job.completed_at or datetime.now(UTC),
                    )
                    for asset in job.generated_assets
                    if asset.source_asset_id is not None
                ]
            )
            await self._session.flush()

    # Claim bằng row lock skip-locked rồi commit ngay trước lời gọi provider dài và có phí.
    async def claim_for_processing(
        self,
        job_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> ImageOptimizationJob | None:
        """Một thời điểm chỉ một worker nhận aggregate; lease hết hạn cho phép recovery."""

        statement = (
            select(ImageOptimizationJobRecord)
            .where(ImageOptimizationJobRecord.job_id == job_id)
            .with_for_update(skip_locked=True)
        )
        record = (await self._session.execute(statement)).scalar_one_or_none()
        if record is None:
            return None
        job = await self._to_domain(record)
        try:
            claimed = job.claim(worker_id=worker_id, lease_seconds=lease_seconds)
        except InvalidJobTransitionError:
            await self._session.rollback()
            return None
        self._copy_job(record, claimed)
        await self._session.commit()
        return claimed

    # Đọc job theo ID và optional owner filter để chặn cross-seller access.
    async def find_by_id(self, job_id: UUID, seller_owner_id: UUID | None = None) -> ImageOptimizationJob | None:
        """Trả aggregate đã map output table hoặc None."""

        statement = select(ImageOptimizationJobRecord).where(ImageOptimizationJobRecord.job_id == job_id)
        if seller_owner_id is not None:
            statement = statement.where(ImageOptimizationJobRecord.seller_owner_id == seller_owner_id)
        record = (await self._session.execute(statement)).scalar_one_or_none()
        return await self._to_domain(record) if record else None

    # Đọc tất cả job theo batch foreign key và thứ tự tạo ổn định.
    async def find_jobs_by_batch(self, batch_id: UUID) -> tuple[ImageOptimizationJob, ...]:
        """Không dùng wildcard idempotency key."""

        statement = (
            select(ImageOptimizationJobRecord)
            .where(ImageOptimizationJobRecord.batch_id == batch_id)
            .order_by(ImageOptimizationJobRecord.created_at.asc())
        )
        records = (await self._session.execute(statement)).scalars().all()
        return tuple([await self._to_domain(record) for record in records])

    # Giữ method legacy bằng cách resolve batch exact-key trước.
    async def find_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationJob | None:
        """Trả job đầu tiên của batch hoặc None."""

        batch = await self.find_batch(seller_owner_id, idempotency_key)
        if batch is None:
            return None
        jobs = await self.find_jobs_by_batch(batch.batch_id)
        return jobs[0] if jobs else None

    # Giữ method compatibility nhưng không còn query LIKE.
    async def find_batch_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> tuple[ImageOptimizationJob, ...]:
        """Trả tuple rỗng khi batch không tồn tại."""

        batch = await self.find_batch(seller_owner_id, idempotency_key)
        return await self.find_jobs_by_batch(batch.batch_id) if batch else ()

    # Dùng SQL COUNT để không tải mọi status row về Python.
    async def count_applied(self, seller_owner_id: UUID) -> int:
        """Đếm job APPLIED của seller bằng database."""

        return await self.count_status(seller_owner_id, ImageOptimizationStatus.APPLIED)

    # Dùng SQL COUNT cho metric theo status.
    async def count_status(self, seller_owner_id: UUID, status: ImageOptimizationStatus) -> int:
        """Trả integer từ aggregate query."""

        statement = (
            select(func.count())
            .select_from(ImageOptimizationJobRecord)
            .where(
                ImageOptimizationJobRecord.seller_owner_id == seller_owner_id,
                ImageOptimizationJobRecord.status == status.value,
            )
        )
        return int((await self._session.execute(statement)).scalar_one())

    # Copy domain snapshot sang ORM record mà không đưa binary/prompt rõ vào database.
    def _copy_job(self, record: ImageOptimizationJobRecord, job: ImageOptimizationJob) -> None:
        """Mirror legacy output JSON tạm thời; output table vẫn là source of truth mới."""

        record.batch_id = job.batch_id
        record.source_asset_ids = [str(value) for value in job.source_asset_ids]
        record.requested_modes = [value.value for value in job.requested_modes]
        record.generation_profile = job.generation_profile.value
        record.selected_output_asset_ids = [str(value) for value in job.selected_output_asset_ids]
        record.generated_asset_ids = [str(value) for value in job.generated_asset_ids]
        record.generated_assets = [
            {
                "output_id": str(value.output_id),
                "source_asset_id": str(value.source_asset_id) if value.source_asset_id else None,
                "asset_id": str(value.asset_id),
                "public_url": value.public_url,
                "mode": value.mode,
            }
            for value in job.generated_assets
        ]
        record.status = job.status.value
        record.request_hash = job.request_hash
        record.provider = job.provider
        record.model = job.model
        record.prompt_version = job.prompt_version
        record.attempt = job.attempt
        record.version = job.version
        record.lease_owner = job.lease_owner
        record.lease_expires_at = job.lease_expires_at
        record.failure_code = job.failure_code
        record.expected_product_updated_at = job.expected_product_updated_at
        record.background_preset = job.background_preset.value if job.background_preset else None
        record.background_description_ciphertext = job.background_description_ciphertext
        record.background_description_hash = job.background_description_hash
        record.processing_stage = job.processing_stage.value
        record.started_at = job.started_at
        record.completed_at = job.completed_at
        record.retention_expires_at = job.retention_expires_at

    # Map batch ORM thành domain immutable.
    @staticmethod
    def _batch_to_domain(record: ImageOptimizationBatchRecord) -> ImageOptimizationBatch:
        """Không trả ORM object ra ngoài infrastructure."""

        return ImageOptimizationBatch(
            batch_id=record.batch_id,
            seller_owner_id=record.seller_owner_id,
            idempotency_key=record.idempotency_key,
            request_hash=record.request_hash,
            created_at=record.created_at,
        )

    # Map output table; với legacy single-source job có thể backfill source mapping an toàn.
    async def _load_outputs(self, record: ImageOptimizationJobRecord) -> tuple[GeneratedAsset, ...]:
        """Legacy multi-source không rõ mapping sẽ không tạo output apply được."""

        statement = select(ImageOptimizationOutputRecord).where(ImageOptimizationOutputRecord.job_id == record.job_id)
        output_records = (await self._session.execute(statement)).scalars().all()
        if output_records:
            legacy_url_by_asset = {
                str(item.get("asset_id")): item.get("public_url")
                for item in record.generated_assets
                if isinstance(item, dict) and item.get("asset_id")
            }
            return tuple(
                GeneratedAsset(
                    output_id=item.output_id,
                    source_asset_id=item.source_asset_id,
                    asset_id=item.asset_id,
                    public_url=legacy_url_by_asset.get(str(item.asset_id)),
                    mode=item.mode,
                    provider=item.provider,
                    model=item.model,
                    prompt_version=item.prompt_version,
                )
                for item in output_records
            )
        if len(record.source_asset_ids) != 1:
            return ()
        source_asset_id = UUID(record.source_asset_ids[0])
        return tuple(
            GeneratedAsset(
                output_id=UUID(str(value.get("output_id"))) if value.get("output_id") else UUID(str(value["asset_id"])),
                source_asset_id=UUID(str(value.get("source_asset_id"))) if value.get("source_asset_id") else source_asset_id,
                asset_id=UUID(str(value["asset_id"])),
                public_url=value.get("public_url"),
                mode=str(value.get("mode", "UNKNOWN")),
                provider=record.provider,
                model=record.model,
                prompt_version=record.prompt_version,
            )
            for value in record.generated_assets
            if isinstance(value, dict) and value.get("asset_id")
        )

    # Map job ORM và output rows về aggregate domain.
    async def _to_domain(self, record: ImageOptimizationJobRecord) -> ImageOptimizationJob:
        """Tập trung conversion để presentation/application không thấy SQLAlchemy."""

        outputs = await self._load_outputs(record)
        return ImageOptimizationJob(
            job_id=record.job_id,
            batch_id=record.batch_id,
            seller_owner_id=record.seller_owner_id,
            product_id=record.product_id,
            source_asset_ids=tuple(UUID(value) for value in record.source_asset_ids),
            requested_modes=tuple(ImageOptimizationMode(value) for value in record.requested_modes),
            generation_profile=ImageGenerationProfile(record.generation_profile or "PREVIEW"),
            selected_output_asset_ids=tuple(UUID(value) for value in (record.selected_output_asset_ids or [])),
            idempotency_key=record.idempotency_key,
            request_hash=record.request_hash,
            expected_product_updated_at=record.expected_product_updated_at,
            background_preset=LifestyleBackgroundPreset(record.background_preset) if record.background_preset else None,
            background_description_ciphertext=record.background_description_ciphertext,
            background_description_hash=record.background_description_hash,
            status=ImageOptimizationStatus(record.status),
            processing_stage=ImageOptimizationProcessingStage(record.processing_stage),
            generated_asset_ids=tuple(asset.asset_id for asset in outputs),
            generated_assets=outputs,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            attempt=record.attempt,
            version=record.version,
            lease_owner=record.lease_owner,
            lease_expires_at=record.lease_expires_at,
            failure_code=record.failure_code,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            retention_expires_at=record.retention_expires_at,
        )
