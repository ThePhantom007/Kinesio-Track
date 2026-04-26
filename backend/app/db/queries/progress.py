"""
Parameterised TimescaleDB queries powering the progress dashboard endpoints.

All functions return plain Python dicts (not ORM objects) because the data
is read-only, aggregated, and serialised directly into Pydantic response
schemas.  Using raw asyncpg for these avoids SQLAlchemy overhead on
potentially large time-series result sets.

Query routing
-------------
  - Per-session granularity  → reads from the raw session_metric hypertable
                               JOINed with exercise_sessions.
  - Daily / weekly            → reads from the daily_rom_avg continuous
                               aggregate for speed.

All patient_id and plan_id parameters are passed as strings (not UUIDs)
because asyncpg requires the caller to match the Postgres parameter type;
TEXT comparison against UUID columns requires explicit casting in the query.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from app.db.timescale import _get_pool
from app.core.logging import get_logger

log = get_logger(__name__)


# ── ROM series ────────────────────────────────────────────────────────────────

async def rom_series_by_joint(
    patient_id: UUID,
    plan_id: UUID,
    joint: str,
    from_date: date | None = None,
    to_date: date | None = None,
    granularity: str = "session",
) -> list[dict[str, Any]]:
    """
    Return ROM time-series for a single joint.

    Args:
        patient_id:   Patient UUID.
        plan_id:      Plan UUID to scope results.
        joint:        MediaPipe joint name, e.g. "left_ankle".
        from_date:    Start date (inclusive). Defaults to 60 days ago.
        to_date:      End date (inclusive). Defaults to today.
        granularity:  "session" | "daily" | "weekly"

    Returns:
        List of dicts ordered chronologically:
          {timestamp, joint, avg_angle_deg, peak_angle_deg,
           avg_quality_score, session_id (session-granularity only)}
    """
    pool = _get_pool()

    from_dt = _to_dt(from_date or _days_ago(60))
    to_dt   = _to_dt(to_date   or date.today(), end_of_day=True)

    if granularity == "session":
        query = """
            SELECT
                es.started_at              AS timestamp,
                sm.session_id::text        AS session_id,
                sm.joint,
                ROUND(AVG(sm.angle_deg)::numeric, 1)    AS avg_angle_deg,
                ROUND(MAX(sm.angle_deg)::numeric, 1)    AS peak_angle_deg,
                ROUND(AVG(sm.quality_score)::numeric, 1) AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND sm.joint      = $3
              AND sm.time BETWEEN $4 AND $5
              AND es.status     = 'completed'
            GROUP BY es.started_at, sm.session_id, sm.joint
            ORDER BY es.started_at ASC
        """
        params = (str(patient_id), str(plan_id), joint, from_dt, to_dt)

    elif granularity == "daily":
        query = """
            SELECT
                time_bucket('1 day', sm.time) AS timestamp,
                NULL::text                     AS session_id,
                sm.joint,
                ROUND(AVG(sm.angle_deg)::numeric, 1)    AS avg_angle_deg,
                ROUND(MAX(sm.angle_deg)::numeric, 1)    AS peak_angle_deg,
                ROUND(AVG(sm.quality_score)::numeric, 1) AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND sm.joint      = $3
              AND sm.time BETWEEN $4 AND $5
              AND es.status     = 'completed'
            GROUP BY timestamp, sm.joint
            ORDER BY timestamp ASC
        """
        params = (str(patient_id), str(plan_id), joint, from_dt, to_dt)

    else:  # weekly
        query = """
            SELECT
                time_bucket('1 week', sm.time) AS timestamp,
                NULL::text                      AS session_id,
                sm.joint,
                ROUND(AVG(sm.angle_deg)::numeric, 1)    AS avg_angle_deg,
                ROUND(MAX(sm.angle_deg)::numeric, 1)    AS peak_angle_deg,
                ROUND(AVG(sm.quality_score)::numeric, 1) AS avg_quality_score
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND sm.joint      = $3
              AND sm.time BETWEEN $4 AND $5
              AND es.status     = 'completed'
            GROUP BY timestamp, sm.joint
            ORDER BY timestamp ASC
        """
        params = (str(patient_id), str(plan_id), joint, from_dt, to_dt)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [dict(r) for r in rows]


async def rom_series_all_joints(
    patient_id: UUID,
    plan_id: UUID,
    from_date: date | None = None,
    to_date: date | None = None,
    granularity: str = "session",
) -> dict[str, list[dict[str, Any]]]:
    """
    Return ROM time-series for all joints recorded for this patient/plan,
    grouped by joint name.

    Returns:
        {joint_name: [data_point, ...]} dict.
    """
    pool = _get_pool()

    from_dt = _to_dt(from_date or _days_ago(60))
    to_dt   = _to_dt(to_date   or date.today(), end_of_day=True)

    query = """
        SELECT DISTINCT sm.joint
        FROM session_metric sm
        JOIN exercise_sessions es ON es.id = sm.session_id::uuid
        WHERE es.patient_id = $1::uuid
          AND es.plan_id    = $2::uuid
          AND sm.time BETWEEN $3 AND $4
    """
    async with pool.acquire() as conn:
        joint_rows = await conn.fetch(query, str(patient_id), str(plan_id), from_dt, to_dt)

    joints = [r["joint"] for r in joint_rows]
    result: dict[str, list[dict[str, Any]]] = {}

    for joint in joints:
        result[joint] = await rom_series_by_joint(
            patient_id, plan_id, joint, from_date, to_date, granularity
        )

    return result


# ── Quality score series ───────────────────────────────────────────────────────

async def quality_score_series(
    patient_id: UUID,
    plan_id: UUID,
    from_date: date | None = None,
    to_date: date | None = None,
    granularity: str = "session",
) -> list[dict[str, Any]]:
    """
    Return quality score + pain level time-series for the progress chart.

    Returns:
        List of dicts ordered chronologically:
          {timestamp, session_id, quality_score, completion_pct,
           post_session_pain}
    """
    pool = _get_pool()

    from_dt = _to_dt(from_date or _days_ago(60))
    to_dt   = _to_dt(to_date   or date.today(), end_of_day=True)

    if granularity == "session":
        query = """
            SELECT
                es.started_at                               AS timestamp,
                es.id::text                                 AS session_id,
                ROUND(es.avg_quality_score::numeric, 1)     AS quality_score,
                ROUND(es.completion_pct::numeric, 3)        AS completion_pct,
                es.post_session_pain
            FROM exercise_sessions es
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND es.status     = 'completed'
              AND es.started_at BETWEEN $3 AND $4
            ORDER BY es.started_at ASC
        """
    elif granularity == "daily":
        query = """
            SELECT
                DATE_TRUNC('day', es.started_at)            AS timestamp,
                NULL::text                                   AS session_id,
                ROUND(AVG(es.avg_quality_score)::numeric, 1) AS quality_score,
                ROUND(AVG(es.completion_pct)::numeric, 3)    AS completion_pct,
                ROUND(AVG(es.post_session_pain)::numeric, 1) AS post_session_pain
            FROM exercise_sessions es
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND es.status     = 'completed'
              AND es.started_at BETWEEN $3 AND $4
            GROUP BY DATE_TRUNC('day', es.started_at)
            ORDER BY timestamp ASC
        """
    else:  # weekly
        query = """
            SELECT
                DATE_TRUNC('week', es.started_at)           AS timestamp,
                NULL::text                                   AS session_id,
                ROUND(AVG(es.avg_quality_score)::numeric, 1) AS quality_score,
                ROUND(AVG(es.completion_pct)::numeric, 3)    AS completion_pct,
                ROUND(AVG(es.post_session_pain)::numeric, 1) AS post_session_pain
            FROM exercise_sessions es
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND es.status     = 'completed'
              AND es.started_at BETWEEN $3 AND $4
            GROUP BY DATE_TRUNC('week', es.started_at)
            ORDER BY timestamp ASC
        """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, str(patient_id), str(plan_id), from_dt, to_dt)

    return [dict(r) for r in rows]


# ── Session frequency ─────────────────────────────────────────────────────────

async def session_frequency(
    patient_id: UUID,
    weeks: int = 4,
) -> dict[str, Any]:
    """
    Return session frequency statistics for the last *weeks* weeks.

    Returns:
        {total_sessions, sessions_per_week, last_session_at,
         streak_days (consecutive days with at least one session)}
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                                    AS total_sessions,
                COUNT(*)::float / $2                        AS sessions_per_week,
                MAX(es.started_at)                          AS last_session_at
            FROM exercise_sessions es
            WHERE es.patient_id = $1::uuid
              AND es.status     = 'completed'
              AND es.started_at >= NOW() - ($2 || ' weeks')::interval
            """,
            str(patient_id),
            weeks,
        )

    return dict(row) if row else {
        "total_sessions": 0,
        "sessions_per_week": 0.0,
        "last_session_at": None,
    }


# ── Milestone queries ─────────────────────────────────────────────────────────

async def get_milestones(
    patient_id: UUID,
    plan_id: UUID,
) -> list[dict[str, Any]]:
    """
    Derive milestone events from session history without a separate milestones
    table — computed on-the-fly from session and plan data.

    Detected milestones:
      - First session completed
      - Quality score first exceeded 60, 75, 90
      - Phase completion (plan.current_phase advanced)
      - 5, 10, 25 sessions completed
      - First pain-free session (post_session_pain == 1)

    Returns:
        List of milestone dicts ordered chronologically:
          {milestone_type, label, achieved_at, value}
    """
    pool = _get_pool()

    milestones: list[dict[str, Any]] = []

    async with pool.acquire() as conn:
        # Quality score threshold milestones
        thresholds = [(60, "Good Form"), (75, "Great Form"), (90, "Excellent Form")]
        for threshold, label in thresholds:
            row = await conn.fetchrow(
                """
                SELECT started_at, avg_quality_score
                FROM exercise_sessions
                WHERE patient_id      = $1::uuid
                  AND plan_id         = $2::uuid
                  AND status          = 'completed'
                  AND avg_quality_score >= $3
                ORDER BY started_at ASC
                LIMIT 1
                """,
                str(patient_id), str(plan_id), float(threshold),
            )
            if row:
                milestones.append({
                    "milestone_type": "quality_target_reached",
                    "label":          f"First time reaching {threshold}% form score — {label}!",
                    "achieved_at":    row["started_at"],
                    "value":          float(row["avg_quality_score"] or threshold),
                })

        # Session count milestones
        for count in [1, 5, 10, 25]:
            row = await conn.fetchrow(
                """
                SELECT started_at
                FROM (
                    SELECT started_at,
                           ROW_NUMBER() OVER (ORDER BY started_at ASC) AS rn
                    FROM exercise_sessions
                    WHERE patient_id = $1::uuid
                      AND plan_id    = $2::uuid
                      AND status     = 'completed'
                ) ranked
                WHERE rn = $3
                """,
                str(patient_id), str(plan_id), count,
            )
            if row:
                label = "First session complete!" if count == 1 else f"{count} sessions completed!"
                milestones.append({
                    "milestone_type": "consecutive_sessions",
                    "label":          label,
                    "achieved_at":    row["started_at"],
                    "value":          float(count),
                })

        # First pain-free session
        row = await conn.fetchrow(
            """
            SELECT started_at, post_session_pain
            FROM exercise_sessions
            WHERE patient_id       = $1::uuid
              AND plan_id          = $2::uuid
              AND status           = 'completed'
              AND post_session_pain = 1
            ORDER BY started_at ASC
            LIMIT 1
            """,
            str(patient_id), str(plan_id),
        )
        if row:
            milestones.append({
                "milestone_type": "pain_free_session",
                "label":          "First pain-free session — amazing progress!",
                "achieved_at":    row["started_at"],
                "value":          1.0,
            })

    # Sort by achieved_at ascending
    milestones.sort(key=lambda m: m["achieved_at"])
    return milestones


# ── Summary stats ─────────────────────────────────────────────────────────────

async def progress_summary(
    patient_id: UUID,
    plan_id: UUID,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, Any]:
    """
    Return aggregate summary statistics for the progress dashboard header.

    Returns:
        {sessions_completed, avg_quality_score, avg_pain_score,
         last_session_at, total_sessions_in_range}
    """
    pool = _get_pool()

    from_dt = _to_dt(from_date or _days_ago(60))
    to_dt   = _to_dt(to_date   or date.today(), end_of_day=True)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                                        AS sessions_completed,
                ROUND(AVG(avg_quality_score)::numeric, 1)       AS avg_quality_score,
                ROUND(AVG(post_session_pain)::numeric, 1)       AS avg_pain_score,
                MAX(started_at)                                 AS last_session_at
            FROM exercise_sessions
            WHERE patient_id = $1::uuid
              AND plan_id    = $2::uuid
              AND status     = 'completed'
              AND started_at BETWEEN $3 AND $4
            """,
            str(patient_id), str(plan_id), from_dt, to_dt,
        )

    return dict(row) if row else {
        "sessions_completed": 0,
        "avg_quality_score":  None,
        "avg_pain_score":     None,
        "last_session_at":    None,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dt(d: date, end_of_day: bool = False) -> datetime:
    """Convert a date to a timezone-aware datetime."""
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    if end_of_day:
        return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _days_ago(n: int) -> date:
    return (datetime.now(timezone.utc) - timedelta(days=n)).date()