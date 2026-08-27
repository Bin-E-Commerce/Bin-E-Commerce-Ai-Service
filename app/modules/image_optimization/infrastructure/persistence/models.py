"""SQLAlchemy mappings cho batch, job, output và outbox tối ưu ảnh.

Các bảng chỉ lưu metadata/asset ID, không lưu binary, prompt rõ, API key hoặc
signed URL. Legacy JSON columns được giữ tạm để đọc dữ liệu cũ một release.
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Base metadata độc lập để AI Service không import entity của service khác.
class Base(DeclarativeBase):
    """Gốc mapping cho Alembic và async SQLAlchemy."""


# Batch lưu idempotency key theo tenant và hash payload đã chuẩn hóa.
class ImageOptimizationBatchRecord(Base):
    """Unique seller/key ngăn retry hoặc concurrent request tạo batch trùng."""

    __tablename__ = "image_optimization_batches"
    __table_args__ = (UniqueConstraint("seller_owner_id", "idempotency_key", name="uq_ai_image_batch_owner_key"),)

    batch_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    seller_owner_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160))
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# Job lưu state machine và lease cho một sản phẩm trong batch.
class ImageOptimizationJobRecord(Base):
    """Không dùng idempotency prefix; quan hệ batch được lưu bằng foreign key."""

    __tablename__ = "image_optimization_jobs"

    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    batch_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("image_optimization_batches.batch_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    seller_owner_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    product_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    source_asset_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    requested_modes: Mapped[list[str]] = mapped_column(JSONB, default=list)
    # Hai cột dưới là legacy compatibility; code mới đọc output table làm source of truth.
    generated_asset_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    generated_assets: Mapped[list[dict[str, str | None]]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(180), index=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    expected_product_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    background_preset: Mapped[str | None] = mapped_column(String(48), nullable=True)
    background_description_ciphertext: Mapped[str | None] = mapped_column(Text, nullable=True)
    background_description_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_stage: Mapped[str] = mapped_column(String(32), default="QUEUED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# Output table giữ mapping một-một giữa ảnh nguồn và ảnh AI đã upload.
class ImageOptimizationOutputRecord(Base):
    """Asset ID là identity lưu trữ; URL chỉ được resolve qua Media Service khi đọc/apply."""

    __tablename__ = "image_optimization_outputs"
    __table_args__ = (UniqueConstraint("job_id", "asset_id", name="uq_ai_image_output_job_asset"),)

    output_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("image_optimization_jobs.job_id", ondelete="CASCADE"),
        index=True,
    )
    source_asset_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    asset_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    mode: Mapped[str] = mapped_column(String(48))
    provider: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# Outbox bảo đảm job và event được ghi cùng transaction trước khi relay sang Kafka.
class ImageOptimizationOutboxRecord(Base):
    """Theo dõi claim/retry/dead-letter để nhiều relay không cùng publish một row."""

    __tablename__ = "image_optimization_outbox_events"

    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    aggregate_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    event_type: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dead_lettered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(80), nullable=True)
