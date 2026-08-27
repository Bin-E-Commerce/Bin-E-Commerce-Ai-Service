"""Thêm batch idempotency, output source mapping, worker lease và outbox retry bền vững."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "202608260002"
down_revision: str | Sequence[str] | None = "202608260001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Bổ sung schema mới theo hướng additive và backfill chỉ những mapping legacy có thể chứng minh đúng.
def upgrade() -> None:
    """Tạo batch/output, lease/version và task retry cho outbox."""

    op.create_table(
        "image_optimization_batches",
        sa.Column("batch_id", sa.Uuid(), primary_key=True),
        sa.Column("seller_owner_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(160), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("seller_owner_id", "idempotency_key", name="uq_ai_image_batch_owner_key"),
    )
    op.create_index("ix_image_optimization_batches_seller_owner_id", "image_optimization_batches", ["seller_owner_id"])
    op.create_index("ix_image_optimization_batches_request_hash", "image_optimization_batches", ["request_hash"])

    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.add_column(sa.Column("batch_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("request_hash", sa.String(64), nullable=True))
        batch.add_column(sa.Column("version", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("lease_owner", sa.String(128), nullable=True))
        batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_foreign_key(
            "fk_ai_image_job_batch",
            "image_optimization_batches",
            ["batch_id"],
            ["batch_id"],
            ondelete="CASCADE",
        )
        batch.create_index("ix_image_optimization_jobs_batch_id", ["batch_id"])

    op.create_table(
        "image_optimization_outputs",
        sa.Column("output_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Uuid(),
            sa.ForeignKey("image_optimization_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("asset_id", sa.Uuid(), nullable=False),
        sa.Column("mode", sa.String(48), nullable=False),
        sa.Column("provider", sa.String(80)),
        sa.Column("model", sa.String(120)),
        sa.Column("prompt_version", sa.String(40)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "asset_id", name="uq_ai_image_output_job_asset"),
    )
    op.create_index("ix_image_optimization_outputs_job_id", "image_optimization_outputs", ["job_id"])
    op.create_index("ix_image_optimization_outputs_asset_id", "image_optimization_outputs", ["asset_id"])
    op.create_index("ix_image_optimization_outputs_source_asset_id", "image_optimization_outputs", ["source_asset_id"])

    # Legacy một source có mapping duy nhất; legacy nhiều source bị đánh dấu để presentation không cho apply.
    op.execute(
        """
        INSERT INTO image_optimization_outputs (
            output_id, job_id, source_asset_id, asset_id, mode, provider, model, prompt_version, created_at
        )
        SELECT
            COALESCE((asset ->> 'output_id')::uuid, (asset ->> 'asset_id')::uuid),
            job.job_id,
            (job.source_asset_ids ->> 0)::uuid,
            (asset ->> 'asset_id')::uuid,
            COALESCE(asset ->> 'mode', 'UNKNOWN'),
            job.provider,
            job.model,
            job.prompt_version,
            COALESCE(job.completed_at, job.created_at)
        FROM image_optimization_jobs AS job
        CROSS JOIN LATERAL jsonb_array_elements(job.generated_assets) AS asset
        WHERE jsonb_array_length(job.source_asset_ids) = 1
          AND asset ? 'asset_id'
        ON CONFLICT (job_id, asset_id) DO NOTHING
        """
    )
    op.execute(
        """
        UPDATE image_optimization_jobs
        SET failure_code = 'LEGACY_SOURCE_MAPPING_UNKNOWN'
        WHERE jsonb_array_length(source_asset_ids) > 1
          AND jsonb_array_length(generated_assets) > 0
          AND status IN ('SUCCEEDED', 'REVIEW_REQUIRED')
        """
    )

    with op.batch_alter_table("image_optimization_outbox_events") as batch:
        batch.add_column(sa.Column("next_attempt_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("locked_by", sa.String(128)))
        batch.add_column(sa.Column("locked_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("dead_lettered_at", sa.DateTime(timezone=True)))
        batch.alter_column("last_error", existing_type=sa.Text(), type_=sa.String(80))


# Downgrade chỉ gỡ phần additive; legacy JSON vẫn đủ để phiên bản cũ đọc job một ảnh.
def downgrade() -> None:
    """Xóa outbox retry, output table, lease và batch."""

    with op.batch_alter_table("image_optimization_outbox_events") as batch:
        batch.alter_column("last_error", existing_type=sa.String(80), type_=sa.Text())
        batch.drop_column("dead_lettered_at")
        batch.drop_column("locked_at")
        batch.drop_column("locked_by")
        batch.drop_column("next_attempt_at")
    op.drop_index("ix_image_optimization_outputs_source_asset_id", table_name="image_optimization_outputs")
    op.drop_index("ix_image_optimization_outputs_asset_id", table_name="image_optimization_outputs")
    op.drop_index("ix_image_optimization_outputs_job_id", table_name="image_optimization_outputs")
    op.drop_table("image_optimization_outputs")
    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.drop_index("ix_image_optimization_jobs_batch_id")
        batch.drop_constraint("fk_ai_image_job_batch", type_="foreignkey")
        batch.drop_column("lease_expires_at")
        batch.drop_column("lease_owner")
        batch.drop_column("version")
        batch.drop_column("request_hash")
        batch.drop_column("batch_id")
    op.drop_index("ix_image_optimization_batches_request_hash", table_name="image_optimization_batches")
    op.drop_index("ix_image_optimization_batches_seller_owner_id", table_name="image_optimization_batches")
    op.drop_table("image_optimization_batches")
