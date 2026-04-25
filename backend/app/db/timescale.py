"""
Raw asyncpg helpers for TimescaleDB hypertable writes and time-range reads.

Why not SQLAlchemy ORM?
-----------------------
The session_metric hypertable receives high-throughput burst writes (dozens
of rows per second during live sessions).  asyncpg's native COPY protocol and
executemany are significantly faster than SQLAlchemy's ORM insert path for
this use case.  All other tables continue to use the ORM.

Hypertable schema (created by Alembic migration 0003_timescale_hypertables.py)
-------------------------------------------------------------------------------
CREATE TABLE session_metric (
    time          TIMESTAMPTZ NOT NULL,
    session_id    UUID        NOT NULL,
    exercise_id   UUID,
    joint         TEXT        NOT NULL,
    angle_deg     FLOAT       NOT NULL,
    quality_score FLOAT
);
SELECT create_hypertable('session_metric', 'time', chunk_time_interval => INTERVAL '1 week');
CREATE INDEX ON session_metric (session_id, time DESC);
CREATE INDEX ON session_metric (joint, time DESC);

Continuous aggregates (also in migration 0003)
----------------------------------------------
CREATE MATERIALIZED VIEW daily_rom_avg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    session_id,
    joint,
    AVG(angle_deg)     AS avg_angle_deg,
    MAX(angle_deg)     AS peak_angle_deg,
    AVG(quality_score) AS avg_quality_score
FROM session_metric
GROUP BY bucket, session_id, joint;
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Module-level asyncpg connection pool — separate from SQLAlchemy's pool.
_pool: asyncpg.Pool | None = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def create_timescale_pool() -> None:
    """
    Create a dedicated asyncpg connection pool for TimescaleDB operations.
    Called from create_db_pool() in postgres.py during app startup.
    """
    global _pool

    # Strip the +asyncpg driver prefix that SQLAlchemy requires but asyncpg rejects.
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
        statement_cache_size=100,
    )
    log.info("timescale_pool_created", min_size=2, max_size=10)


async def close_timescale_pool() -> None:
    """Close the asyncpg pool. Called during app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("timescale_pool_closed")


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError(
            "TimescaleDB pool not initialised. Call create_timescale_pool() first."
        )
    return _pool


# ── Hypertable setup ──────────────────────────────────────────────────────────

async def create_hypertable() -> None:
    """
    Ensure the session_metric hypertable exists.
    Idempotent — called at startup as a safety net alongside the Alembic migration.
    """
    pool = _get_pool()
    async with pool.acquire() as conn:
        # create_hypertable raises if the table doesn't exist; migration handles creation.
        try:
            await conn.execute(
                """
                SELECT create_hypertable(
                    'session_metric', 'time',
                    chunk_time_interval => INTERVAL '1 week',
                    if_not_exists => TRUE
                );
                """
            )
            log.info("session_metric_hypertable_verified")
        except asyncpg.exceptions.UndefinedTableError:
            log.warning(
                "hypertable_table_missing",
                hint="Run Alembic migrations first: alembic upgrade head",
            )
        except Exception as exc:
            log.warning("hypertable_create_skipped", reason=str(exc))


# ── Writes ────────────────────────────────────────────────────────────────────

async def write_metric_batch(rows: list[dict[str, Any]]) -> None:
    """
    Bulk-insert session metric rows into the hypertable using asyncpg's
    fast executemany path.

    Each row dict must have:
        session_id:    str UUID
        exercise_id:   str UUID | None
        joint:         str (MediaPipe joint name)
        angle_deg:     float
        quality_score: float | None

    The 'time' column is always set to now(UTC) server-side.

    Args:
        rows: List of metric dicts. Empty list is a no-op.
    """
    if not rows:
        return

    pool = _get_pool()
    now  = datetime.now(timezone.utc)

    records = [
        (
            now,
            row["session_id"],
            row.get("exercise_id"),
            row["joint"],
            float(row["angle_deg"]),
            float(row["quality_score"]) if row.get("quality_score") is not None else None,
        )
        for row in rows
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO session_metric (time, session_id, exercise_id, joint, angle_deg, quality_score)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            records,
        )

    log.debug("timescale_batch_written", row_count=len(records))


# ── Reads ─────────────────────────────────────────────────────────────────────

async def get_rom_series(
    patient_id: UUID,
    plan_id: UUID,
    joint: str | None = None,
    from_dt: datetime | None = None,
    to_dt:   datetime | None = None,
    granularity: str = "session",
) -> list[dict[str, Any]]:
    """
    Return ROM time-series data for a patient and plan.

    Queries the daily_rom_avg continuous aggregate for daily/weekly granularity,
    or the raw hypertable for per-session granularity.

    Args:
        patient_id:   Patient UUID (used to JOIN with exercise_sessions).
        plan_id:      Plan UUID to scope the results.
        joint:        Optional joint name filter, e.g. "left_ankle".
        from_dt:      Start of date range (inclusive). Defaults to 30 days ago.
        to_dt:        End of date range (inclusive). Defaults to now.
        granularity:  "session" | "daily" | "weekly"

    Returns:
        List of dicts with keys: timestamp, joint, avg_angle_deg, peak_angle_deg,
        avg_quality_score, session_id (for session granularity).
    """
    pool = _get_pool()

    from_dt = from_dt or _days_ago(30)
    to_dt   = to_dt   or datetime.now(timezone.utc)

    joint_filter = "AND sm.joint = $5" if joint else ""
    joint_param  = [joint] if joint else []

    if granularity == "session":
        query = f"""
            SELECT
                es.started_at          AS timestamp,
                sm.session_id::text    AS session_id,
                sm.joint,
                AVG(sm.angle_deg)      AS avg_angle_deg,
                MAX(sm.angle_deg)      AS peak_angle_deg,
                AVG(sm.quality_score)  AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1
              AND es.plan_id    = $2
              AND sm.time BETWEEN $3 AND $4
              {joint_filter}
            GROUP BY es.started_at, sm.session_id, sm.joint
            ORDER BY es.started_at ASC
        """
    elif granularity == "daily":
        query = f"""
            SELECT
                time_bucket('1 day', sm.time) AS timestamp,
                NULL::text                     AS session_id,
                sm.joint,
                AVG(sm.angle_deg)              AS avg_angle_deg,
                MAX(sm.angle_deg)              AS peak_angle_deg,
                AVG(sm.quality_score)          AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1
              AND es.plan_id    = $2
              AND sm.time BETWEEN $3 AND $4
              {joint_filter}
            GROUP BY timestamp, sm.joint
            ORDER BY timestamp ASC
        """
    else:  # weekly
        query = f"""
            SELECT
                time_bucket('1 week', sm.time) AS timestamp,
                NULL::text                      AS session_id,
                sm.joint,
                AVG(sm.angle_deg)               AS avg_angle_deg,
                MAX(sm.angle_deg)               AS peak_angle_deg,
                AVG(sm.quality_score)           AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1
              AND es.plan_id    = $2
              AND sm.time BETWEEN $3 AND $4
              {joint_filter}
            GROUP BY timestamp, sm.joint
            ORDER BY timestamp ASC
        """

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            query,
            str(patient_id),
            str(plan_id),
            from_dt,
            to_dt,
            *joint_param,
        )

    return [dict(row) for row in rows]


async def get_quality_trend(
    patient_id: UUID,
    plan_id: UUID,
    n_sessions: int = 10,
) -> list[dict[str, Any]]:
    """
    Return per-session average quality scores in chronological order.
    Used by recovery_forecaster and plan_adapter for trend analysis.

    Returns:
        List of dicts: {session_id, started_at, avg_quality_score, post_session_pain}.
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                es.id::text         AS session_id,
                es.started_at,
                es.avg_quality_score,
                es.post_session_pain
            FROM exercise_sessions es
            WHERE es.patient_id = $1
              AND es.plan_id    = $2
              AND es.status     = 'completed'
            ORDER BY es.started_at DESC
            LIMIT $3
            """,
            str(patient_id),
            str(plan_id),
            n_sessions,
        )

    # Return in ascending order so callers can iterate chronologically.
    return list(reversed([dict(row) for row in rows]))


async def get_session_frequency(
    patient_id: UUID,
    weeks: int = 4,
) -> float:
    """
    Return average sessions per week for a patient over the last *weeks* weeks.

    Returns:
        Float sessions-per-week. Returns 0.0 if no sessions found.
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            SELECT COUNT(*)::float / $2
            FROM exercise_sessions
            WHERE patient_id = $1
              AND status     = 'completed'
              AND started_at >= NOW() - ($2 || ' weeks')::interval
            """,
            str(patient_id),
            weeks,
        )

    return float(result or 0.0)


# ── Continuous aggregate refresh ───────────────────────────────────────────────

async def refresh_continuous_aggregates() -> None:
    """
    Manually trigger a refresh of all continuous aggregates.
    Called by the Celery beat analytics_tasks.refresh_recovery_estimates task.
    TimescaleDB refreshes aggregates automatically, but manual refresh ensures
    up-to-date data for the daily analytics run.
    """
    pool = _get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                CALL refresh_continuous_aggregate('daily_rom_avg', NULL, NULL);
                """
            )
            log.info("continuous_aggregates_refreshed")
        except Exception as exc:
            log.warning("continuous_aggregate_refresh_failed", error=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> datetime:
    from datetime import timedelta
    return datetime.now(timezone.utc) - timedelta(days=n)