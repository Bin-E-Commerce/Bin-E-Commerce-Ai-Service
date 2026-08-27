"""Baseline schema legacy cho image optimization trước khi chuẩn hóa batch/output."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "202608260001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Tạo schema legacy tương đương ba migration SQL cũ để database mới có thể nâng tuần tự.
def upgrade() -> None:
    """Tạo bảng job/outbox và các cột background đã tồn tại trước refactor."""

    op.create_table(
        "image_optimization_jobs",
        sa.Column("job_id", sa.Uuid(), primary_key=True),
        sa.Column("seller_owner_id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("requested_modes", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("generated_asset_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("generated_assets", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(180), nullable=False, unique=True),
        sa.Column("provider", sa.String(80)),
        sa.Column("model", sa.String(120)),
        sa.Column("prompt_version", sa.String(40)),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_code", sa.String(80)),
        sa.Column("expected_product_updated_at", sa.DateTime(timezone=True)),
        sa.Column("background_preset", sa.String(48)),
        sa.Column("background_description_ciphertext", sa.Text()),
        sa.Column("background_description_hash", sa.String(64)),
        sa.Column("processing_stage", sa.String(32), nullable=False, server_default="QUEUED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_image_optimization_jobs_seller_status", "image_optimization_jobs", ["seller_owner_id", "status"])
    op.create_table(
        "image_optimization_outbox_events",
        sa.Column("event_id", sa.Uuid(), primary_key=True),
        sa.Column("aggregate_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(120), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
    )
    op.create_index(
        "idx_image_optimization_outbox_unpublished",
        "image_optimization_outbox_events",
        ["published_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


# Xóa schema legacy theo thứ tự ngược quan hệ sử dụng.
def downgrade() -> None:
    """Gỡ bảng outbox rồi bảng job."""

    op.drop_index("idx_image_optimization_outbox_unpublished", table_name="image_optimization_outbox_events")
    op.drop_table("image_optimization_outbox_events")
    op.drop_index("idx_image_optimization_jobs_seller_status", table_name="image_optimization_jobs")
    op.drop_table("image_optimization_jobs")
