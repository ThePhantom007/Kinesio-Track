"""
Alembic migration environment — configured for async SQLAlchemy with asyncpg.

Key behaviours
--------------
  - Reads DATABASE_URL from app/core/config.py (same source as the app).
  - Imports every ORM model via ``app.models`` so autogenerate detects all
    table additions and changes without manual include_schemas configuration.
  - Uses ``run_async_migrations()`` for the online mode so migrations run
    through asyncpg rather than a synchronous psycopg2 driver.
  - Offline mode (generating SQL scripts without a live DB) works normally
    and uses the URL directly.

Running migrations
------------------
  # Apply all pending migrations
  alembic upgrade head

  # Generate a new migration from model changes
  alembic revision --autogenerate -m "describe the change"

  # Show current revision
  alembic current

  # Roll back one step
  alembic downgrade -1
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Alembic config object ─────────────────────────────────────────────────────

config = context.config

# Interpret the config file's logging section if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import all models so autogenerate sees every table ────────────────────────
# This must come before target_metadata is set.

from app.models import Base  # noqa: E402  (imports all models transitively)

target_metadata = Base.metadata

# ── Inject DATABASE_URL from app settings ─────────────────────────────────────
# Override whatever is in alembic.ini so there is a single source of truth.

from app.core.config import settings  # noqa: E402

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


# ── Offline migrations (generate SQL without live DB) ─────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Emits SQL to stdout without opening a real database connection.
    Useful for generating a migration script to review or apply manually.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render ALTER TABLE ... SET NOT NULL etc. in addition to schema diffs
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (run against live DB) ───────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Include schema-level objects (sequences, etc.)
        include_schemas=True,
        # Render the server_default diff for columns
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine from the Alembic config and run migrations
    through a synchronous connection wrapper (required by Alembic's
    context API, which is not natively async).
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # no pool needed for one-off migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — runs the async function."""
    asyncio.run(run_async_migrations())


# ── Dispatch ──────────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()