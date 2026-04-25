"""
Async SQLAlchemy engine, session factory, and startup initialisation.

Engine configuration
--------------------
Uses asyncpg as the driver (DATABASE_URL must be postgresql+asyncpg://).
Pool sizing follows the rule of thumb: pool_size ≈ (CPU cores × 2) + 1,
overridable via DB_POOL_SIZE in config.

Lifespan integration
--------------------
create_db_pool() and close_db_pool() are called from the FastAPI lifespan
context manager in app/main.py.  The engine and session factory are stored
on app.state so route handlers can access them via the get_db() dependency.

TimescaleDB initialisation
--------------------------
init_timescaledb() is called once at startup to ensure the TimescaleDB
extension is enabled and the rep_metrics hypertable exists.  It is
idempotent — safe to call on every startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Module-level singletons — populated by create_db_pool().
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def create_db_pool() -> None:
    """
    Create the async SQLAlchemy engine and session factory.
    Called once during FastAPI app startup.
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,          # validate connections before checkout
        pool_recycle=3600,           # recycle connections every 1 h
        echo=settings.DB_ECHO,
        future=True,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,      # keep attributes accessible after commit
        autoflush=False,             # explicit flush control in services
        autocommit=False,
    )

    log.info(
        "db_pool_created",
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        url=_sanitised_url(settings.DATABASE_URL),
    )

    await init_timescaledb()


async def close_db_pool() -> None:
    """Dispose the engine connection pool. Called during FastAPI app shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        log.info("db_pool_closed")


# ── Session dependency ─────────────────────────────────────────────────────────

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a transactional AsyncSession.
    Commits on clean exit, rolls back on any exception.

    Use this in Celery tasks and scripts where FastAPI Depends() is unavailable.
    """
    if _session_factory is None:
        raise RuntimeError("DB pool has not been initialised. Call create_db_pool() first.")

    async with _session_factory() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a transactional AsyncSession.

    Usage::

        @router.post("/example")
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...

    Commits on clean exit, rolls back on any exception, closes on exit.
    """
    if _session_factory is None:
        raise RuntimeError("DB pool has not been initialised.")

    async with _session_factory() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise


# ── Engine accessor (for raw asyncpg access in timescale.py) ──────────────────

def get_engine() -> AsyncEngine:
    """Return the current engine. Raises if pool not initialised."""
    if _engine is None:
        raise RuntimeError("DB pool has not been initialised.")
    return _engine


# ── TimescaleDB initialisation ────────────────────────────────────────────────

async def init_timescaledb() -> None:
    """
    Ensure the TimescaleDB extension is loaded and the session_metric
    hypertable exists.  Idempotent — safe to call on every startup.

    The hypertable creation is handled by Alembic migration
    0003_timescale_hypertables.py, but we also verify it here so local
    development works after a fresh docker-compose up without running
    migrations manually.
    """
    if _engine is None:
        return

    try:
        async with _engine.connect() as conn:
            # Enable extension (no-op if already enabled)
            await conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            )
            await conn.commit()
            log.info("timescaledb_extension_verified")
    except Exception as exc:
        # TimescaleDB may not be available in plain Postgres test environments.
        log.warning(
            "timescaledb_init_skipped",
            reason=str(exc),
            hint="Run with the timescaledb Docker image for full functionality.",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitised_url(url: str) -> str:
    """Strip the password from a DB URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url