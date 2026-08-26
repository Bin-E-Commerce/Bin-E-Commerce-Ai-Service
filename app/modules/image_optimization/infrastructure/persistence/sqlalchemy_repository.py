"""SQLAlchemy repository adapter map aggregate image optimization sang PostgreSQL."""

from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.image_optimization.domain.enums import (
    ImageOptimizationMode,
    ImageOptimizationProcessingStage,
    ImageOptimizationStatus,
    LifestyleBackgroundPreset,
)
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob
from app.modules.image_optimization.infrastructure.persistence.models import ImageOptimizationJobRecord


class SqlAlchemyImageOptimizationJobRepository:
    """Repository production luu job trong transaction session do composition layer cung cap."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, job: ImageOptimizationJob) -> None:
        """Upsert record va flush de outbox cung transaction co the commit atomically."""

        record = await self._session.get(ImageOptimizationJobRecord, job.job_id)
        if record is None:
            record = ImageOptimizationJobRecord(
                job_id=job.job_id,
                seller_owner_id=job.seller_owner_id,
                product_id=job.product_id,
                idempotency_key=job.idempotency_key,
                created_at=job.created_at,
            )
            self._session.add(record)
        record.source_asset_ids = [str(value) for value in job.source_asset_ids]
        record.requested_modes = [value.value for value in job.requested_modes]
        record.generated_asset_ids = [str(value) for value in job.generated_asset_ids]
        record.generated_assets = [
            {"asset_id": str(value.asset_id), "public_url": value.public_url, "mode": value.mode}
            for value in job.generated_assets
        ]
        record.status = job.status.value
        record.provider = job.provider
        record.model = job.model
        record.prompt_version = job.prompt_version
        record.attempt = job.attempt
        record.failure_code = job.failure_code
        record.expected_product_updated_at = job.expected_product_updated_at
        record.background_preset = job.background_preset.value if job.background_preset else None
        record.background_description_ciphertext = job.background_description_ciphertext
        record.background_description_hash = job.background_description_hash
        record.processing_stage = job.processing_stage.value
        record.started_at = job.started_at
        record.completed_at = job.completed_at
        record.retention_expires_at = job.retention_expires_at
        await self._session.flush()

    async def find_by_id(self, job_id: UUID, seller_owner_id: UUID | None = None) -> ImageOptimizationJob | None:
        """Doc record theo owner va map lai aggregate khong dua SQLAlchemy vao domain."""

        statement = select(ImageOptimizationJobRecord).where(ImageOptimizationJobRecord.job_id == job_id)
        if seller_owner_id is not None:
            statement = statement.where(ImageOptimizationJobRecord.seller_owner_id == seller_owner_id)
        record = (await self._session.execute(statement)).scalar_one_or_none()
        return self._to_domain(record) if record else None

    async def find_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationJob | None:
        """Tim request cu theo seller va idempotency key de xu ly retry an toan."""

        statement = select(ImageOptimizationJobRecord).where(
            ImageOptimizationJobRecord.seller_owner_id == seller_owner_id,
            or_(
                ImageOptimizationJobRecord.idempotency_key == idempotency_key,
                ImageOptimizationJobRecord.idempotency_key.like(f"{idempotency_key}:%"),
            ),
        )
        record = (await self._session.execute(statement)).scalar_one_or_none()
        return self._to_domain(record) if record else None

    async def count_applied(self, seller_owner_id: UUID) -> int:
        """Dem job APPLIED cho metric ma khong can load binary/media."""

        records = (
            (
                await self._session.execute(
                    select(ImageOptimizationJobRecord.status).where(ImageOptimizationJobRecord.seller_owner_id == seller_owner_id)
                )
            )
            .scalars()
            .all()
        )
        return sum(1 for status in records if status == ImageOptimizationStatus.APPLIED.value)

    async def find_batch_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> tuple[ImageOptimizationJob, ...]:
        """Lay toan bo job cua batch theo prefix idempotency."""

        statement = (
            select(ImageOptimizationJobRecord)
            .where(
                ImageOptimizationJobRecord.seller_owner_id == seller_owner_id,
                ImageOptimizationJobRecord.idempotency_key.like(f"{idempotency_key}%"),
            )
            .order_by(ImageOptimizationJobRecord.created_at.asc())
        )
        records = (await self._session.execute(statement)).scalars().all()
        return tuple(self._to_domain(record) for record in records)

    async def count_status(self, seller_owner_id: UUID, status: ImageOptimizationStatus) -> int:
        """Dem state tai database ma khong load toan bo aggregate."""

        statement = select(ImageOptimizationJobRecord.job_id).where(
            ImageOptimizationJobRecord.seller_owner_id == seller_owner_id,
            ImageOptimizationJobRecord.status == status.value,
        )
        return len((await self._session.execute(statement)).scalars().all())

    def _to_domain(self, record: ImageOptimizationJobRecord) -> ImageOptimizationJob:
        """Map persistence primitive ve aggregate immutable cua domain."""

        return ImageOptimizationJob(
            job_id=record.job_id,
            seller_owner_id=record.seller_owner_id,
            product_id=record.product_id,
            source_asset_ids=tuple(UUID(value) for value in record.source_asset_ids),
            requested_modes=tuple(ImageOptimizationMode(value) for value in record.requested_modes),
            idempotency_key=record.idempotency_key,
            expected_product_updated_at=record.expected_product_updated_at,
            background_preset=LifestyleBackgroundPreset(record.background_preset) if record.background_preset else None,
            background_description_ciphertext=record.background_description_ciphertext,
            background_description_hash=record.background_description_hash,
            status=ImageOptimizationStatus(record.status),
            processing_stage=ImageOptimizationProcessingStage(record.processing_stage),
            generated_asset_ids=tuple(UUID(value) for value in record.generated_asset_ids),
            generated_assets=tuple(
                GeneratedAsset(
                    asset_id=UUID(str(value["asset_id"])),
                    public_url=value.get("public_url"),
                    mode=str(value.get("mode", "UNKNOWN")),
                )
                for value in record.generated_assets
                if isinstance(value, dict) and value.get("asset_id")
            ),
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            attempt=record.attempt,
            failure_code=record.failure_code,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            retention_expires_at=record.retention_expires_at,
        )
