"""Cấu hình Alembic async và lấy DATABASE_URL từ typed settings của AI Service."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings
from app.modules.image_optimization.infrastructure.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# Cấu hình migration offline để CI có thể render SQL mà không mở kết nối database.
def run_migrations_offline() -> None:
    """Render SQL theo PostgreSQL dialect với literal bind."""

    settings = get_settings()
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required for Alembic")
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# Chạy migration trong connection async nhưng giao sync connection cho Alembic context.
def _run_migrations(connection: object) -> None:
    """Cấu hình metadata và bật kiểm tra thay đổi kiểu cột."""

    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


# Mở engine NullPool để process migration không giữ connection sau khi hoàn tất.
async def run_migrations_online() -> None:
    """Upgrade/downgrade schema qua SQLAlchemy async engine."""

    settings = get_settings()
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required for Alembic")
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = settings.database_url
    engine = async_engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
