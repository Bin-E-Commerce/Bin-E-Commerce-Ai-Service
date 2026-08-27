"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | Sequence[str] | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


# Áp dụng thay đổi schema của revision này.
def upgrade() -> None:
    """Nâng schema lên revision hiện tại."""

    ${upgrades if upgrades else "pass"}


# Hoàn tác thay đổi schema của revision này.
def downgrade() -> None:
    """Hạ schema về revision trước."""

    ${downgrades if downgrades else "pass"}
