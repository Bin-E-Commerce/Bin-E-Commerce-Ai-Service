"""Thêm mốc analytics để chỉ ảnh đại diện tạo phiên đo lường mới."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "202609180001"
down_revision: str | Sequence[str] | None = "202608270001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Thêm cờ additive và coi các job lifecycle cũ là mốc hợp lệ để không làm mất baseline hiện tại.
def upgrade() -> None:
    """Lưu việc apply có thay đổi ảnh đại diện hay chỉ thay đổi ảnh phụ."""

    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.add_column(sa.Column("starts_impact_session", sa.Boolean(), nullable=False, server_default=sa.text("false")))

    # Dữ liệu cũ chưa có thông tin cover tại thời điểm apply; giữ hành vi analytics cũ cho các job đã hoàn tất.
    op.execute(
        sa.text(
            """
            UPDATE image_optimization_jobs
            SET starts_impact_session = true
            WHERE status IN ('APPLIED', 'ROLLED_BACK')
            """
        )
    )


# Xóa cờ mới mà không đụng tới job, output hoặc các bảng event analytics.
def downgrade() -> None:
    """Gỡ metadata mốc analytics nếu rollback migration."""

    with op.batch_alter_table("image_optimization_jobs") as batch:
        batch.drop_column("starts_impact_session")
