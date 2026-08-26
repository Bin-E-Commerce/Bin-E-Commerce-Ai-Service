"""SQLAlchemy mapping cho metadata job, khong luu binary image hay prompt raw."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base rieng de AI Service khong phu thuoc entity cua Product Service."""


class ImageOptimizationJobRecord(Base):
    """Bang persistence luu state machine va tham chieu asset."""

    __tablename__ = "image_optimization_jobs"

    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    seller_owner_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    product_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    source_asset_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    requested_modes: Mapped[list[str]] = mapped_column(JSONB, default=list)
    generated_asset_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    generated_assets: Mapped[list[dict[str, str | None]]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(180), unique=True)
    provider: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
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


class ImageOptimizationOutboxRecord(Base):
    """Outbox event de khong mat Kafka message giua database commit va publish."""

    __tablename__ = "image_optimization_outbox_events"

    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    aggregate_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    event_type: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
