"""Bổ sung hồ sơ preview/final và lựa chọn output cho image optimization."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "202608270001"
down_revision: str | Sequence[str] | None = "202608260002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Thêm cột additive để job cũ tiếp tục chạy ở profile preview an toàn.
def upgrade() -> None:
    """Lưu profile generation và các output seller chọn để finalization idempotent."""

    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.add_column(sa.Column("generation_profile", sa.String(16), nullable=False, server_default="PREVIEW"))
        batch.add_column(
            sa.Column("selected_output_asset_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb"))
        )


# Xóa đúng phần schema được migration này thêm vào.
def downgrade() -> None:
    """Gỡ profile và lựa chọn output mà không tác động dữ liệu job hiện hữu khác."""

    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.drop_column("selected_output_asset_ids")
        batch.drop_column("generation_profile")
