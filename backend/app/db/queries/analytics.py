"""
Aggregation queries used by the plan_adapter and recovery_forecaster services.

These queries are intentionally separate from db/queries/progress.py because
they serve internal business logic (adaptation decisions, ETA regression) rather
than the external API/dashboard.  They return lightweight dicts, not full
schema objects, to keep the services that call them loosely coupled from the
DB layer.

All functions use the raw asyncpg pool from db/timescale.py for speed.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.db.timescale import _get_pool
from app.core.logging import get_logger

log = get_logger(__name__)


# ── Session metrics ───────────────────────────────────────────────────────────

async def last_n_session_metrics(
    patient_id: UUID,
    plan_id: UUID,
    n: int = 10,
) -> list[dict[str, Any]]:
    """
    Return the last *n* completed sessions' key metrics in chronological order.
    Used by plan_adapter (adaptation decision) and recovery_forecaster (ETA).

    Args:
        patient_id: Patient UUID.
        plan_id:    Plan UUID — scopes results to one treatment course.
        n:          Maximum number of sessions to return.

    Returns:
        List of dicts ordered by started_at ASC:
          {session_id, session_date, avg_quality_score, post_session_pain,
           completion_pct, peak_rom_degrees, duration_seconds}
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id::text                                    AS session_id,
                started_at                                  AS session_date,
                avg_quality_score,
                post_session_pain,
                completion_pct,
                peak_rom_degrees,
                EXTRACT(EPOCH FROM (ended_at - started_at))::int AS duration_seconds
            FROM exercise_sessions
            WHERE patient_id = $1::uuid
              AND plan_id    = $2::uuid
              AND status     = 'completed'
            ORDER BY started_at DESC
            LIMIT $3
            """,
            str(patient_id), str(plan_id), n,
        )

    # Reverse so result is chronological (oldest → newest).
    return list(reversed([dict(r) for r in rows]))


async def last_n_session_metrics_for_exercise(
    patient_id: UUID,
    exercise_id: UUID,
    n: int = 5,
) -> list[dict[str, Any]]:
    """
    Return per-exercise session metrics for a specific exercise.
    Used by plan_adapter when deciding whether to change a specific exercise's
    parameters rather than the whole plan.

    Returns:
        List of dicts ordered ASC: {session_id, session_date,
        avg_quality_score, post_session_pain, peak_rom_degrees}
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                es.id::text                                 AS session_id,
                es.started_at                               AS session_date,
                es.avg_quality_score,
                es.post_session_pain,
                es.peak_rom_degrees
            FROM exercise_sessions es
            WHERE es.patient_id  = $1::uuid
              AND es.exercise_id = $2::uuid
              AND es.status      = 'completed'
            ORDER BY es.started_at DESC
            LIMIT $3
            """,
            str(patient_id), str(exercise_id), n,
        )

    return list(reversed([dict(r) for r in rows]))


# ── Quality trend ─────────────────────────────────────────────────────────────

async def quality_trend_slope(
    patient_id: UUID,
    plan_id: UUID,
    n: int = 10,
) -> dict[str, Any]:
    """
    Compute the linear trend of quality scores over the last *n* sessions
    using Postgres's built-in regression functions (no Python math needed).

    Uses REGR_SLOPE(y, x) where x = session ordinal (1, 2, 3…), y = quality_score.

    Returns:
        {slope, intercept, r_squared, session_count, current_quality,
         trend}  — trend is "improving" | "plateauing" | "regressing"
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH ordered AS (
                SELECT
                    avg_quality_score                               AS quality,
                    ROW_NUMBER() OVER (ORDER BY started_at ASC)    AS rn
                FROM exercise_sessions
                WHERE patient_id = $1::uuid
                  AND plan_id    = $2::uuid
                  AND status     = 'completed'
                ORDER BY started_at DESC
                LIMIT $3
            )
            SELECT
                REGR_SLOPE(quality, rn)         AS slope,
                REGR_INTERCEPT(quality, rn)     AS intercept,
                REGR_R2(quality, rn)            AS r_squared,
                COUNT(*)                        AS session_count,
                MAX(quality)                    AS peak_quality,
                (array_agg(quality ORDER BY rn DESC))[1] AS current_quality
            FROM ordered
            WHERE quality IS NOT NULL
            """,
            str(patient_id), str(plan_id), n,
        )

    if not row or not row["session_count"]:
        return {
            "slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
            "session_count": 0, "current_quality": None,
            "peak_quality": None, "trend": "plateauing",
        }

    slope = float(row["slope"] or 0.0)
    trend = (
        "improving"  if slope >  0.5 else
        "regressing" if slope < -0.5 else
        "plateauing"
    )

    return {
        "slope":           round(slope, 3),
        "intercept":       round(float(row["intercept"] or 0.0), 3),
        "r_squared":       round(float(row["r_squared"] or 0.0), 3),
        "session_count":   int(row["session_count"]),
        "current_quality": float(row["current_quality"]) if row["current_quality"] else None,
        "peak_quality":    float(row["peak_quality"]) if row["peak_quality"] else None,
        "trend":           trend,
    }


# ── Pain trend ────────────────────────────────────────────────────────────────

async def pain_trend(
    patient_id: UUID,
    plan_id: UUID,
    n: int = 5,
) -> dict[str, Any]:
    """
    Return recent pain score statistics.
    Used by red_flag_monitor to detect a sustained pain escalation pattern.

    Returns:
        {avg_pain, max_pain, min_pain, sessions_with_high_pain,
         pain_increasing (bool)}
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH recent AS (
                SELECT
                    post_session_pain,
                    ROW_NUMBER() OVER (ORDER BY started_at ASC) AS rn
                FROM exercise_sessions
                WHERE patient_id = $1::uuid
                  AND plan_id    = $2::uuid
                  AND status     = 'completed'
                  AND post_session_pain IS NOT NULL
                ORDER BY started_at DESC
                LIMIT $3
            )
            SELECT
                ROUND(AVG(post_session_pain)::numeric, 1)   AS avg_pain,
                MAX(post_session_pain)                      AS max_pain,
                MIN(post_session_pain)                      AS min_pain,
                COUNT(*) FILTER (WHERE post_session_pain >= 7) AS sessions_with_high_pain,
                REGR_SLOPE(post_session_pain, rn) > 0.3    AS pain_increasing
            FROM recent
            """,
            str(patient_id), str(plan_id), n,
        )

    if not row:
        return {
            "avg_pain": None, "max_pain": None, "min_pain": None,
            "sessions_with_high_pain": 0, "pain_increasing": False,
        }

    return {
        "avg_pain":                 float(row["avg_pain"] or 0),
        "max_pain":                 int(row["max_pain"] or 0),
        "min_pain":                 int(row["min_pain"] or 0),
        "sessions_with_high_pain":  int(row["sessions_with_high_pain"] or 0),
        "pain_increasing":          bool(row["pain_increasing"]),
    }


# ── ROM comparison ────────────────────────────────────────────────────────────

async def rom_vs_baseline(
    patient_id: UUID,
    plan_id: UUID,
    joint: str,
) -> dict[str, Any]:
    """
    Compare the patient's most recent ROM to their intake baseline for a joint.
    Used by recovery_forecaster to compute improvement percentage.

    Returns:
        {baseline_angle_deg, current_angle_deg, improvement_deg,
         improvement_pct, target_angle_deg}
        Values are None if insufficient data.
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        # Most recent peak ROM for this joint
        current_row = await conn.fetchrow(
            """
            SELECT MAX(sm.angle_deg) AS current_angle_deg
            FROM session_metric sm
            JOIN exercise_sessions es ON es.id = sm.session_id::uuid
            WHERE es.patient_id = $1::uuid
              AND es.plan_id    = $2::uuid
              AND sm.joint      = $3
              AND es.status     = 'completed'
              AND es.started_at >= NOW() - INTERVAL '7 days'
            """,
            str(patient_id), str(plan_id), joint,
        )

        # Baseline ROM from patient profile
        baseline_row = await conn.fetchrow(
            """
            SELECT baseline_rom -> $2 ->> 'angle_deg' AS baseline_angle
            FROM patient_profiles
            WHERE id = $1::uuid
            """,
            str(patient_id), joint,
        )

    current = (
        float(current_row["current_angle_deg"])
        if current_row and current_row["current_angle_deg"] is not None
        else None
    )
    baseline = (
        float(baseline_row["baseline_angle"])
        if baseline_row and baseline_row["baseline_angle"] is not None
        else None
    )

    if current is None or baseline is None or baseline == 0:
        return {
            "baseline_angle_deg": baseline,
            "current_angle_deg":  current,
            "improvement_deg":    None,
            "improvement_pct":    None,
        }

    improvement_deg = current - baseline
    improvement_pct = round((improvement_deg / baseline) * 100, 1)

    return {
        "baseline_angle_deg": round(baseline, 1),
        "current_angle_deg":  round(current, 1),
        "improvement_deg":    round(improvement_deg, 1),
        "improvement_pct":    improvement_pct,
    }


# ── Token usage ───────────────────────────────────────────────────────────────

async def monthly_token_spend(year: int, month: int) -> list[dict[str, Any]]:
    """
    Return per-call-type token spend for a given calendar month.
    Used by the /admin/ai-costs endpoint and the Celery budget-check task.

    Returns:
        List of dicts: {call_type, input_tokens, output_tokens,
        cost_usd, call_count}
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                call_type,
                SUM(input_tokens)::bigint   AS input_tokens,
                SUM(output_tokens)::bigint  AS output_tokens,
                ROUND(SUM(cost_usd)::numeric, 4) AS cost_usd,
                COUNT(*)                    AS call_count
            FROM token_usage
            WHERE EXTRACT(YEAR  FROM called_at) = $1
              AND EXTRACT(MONTH FROM called_at) = $2
            GROUP BY call_type
            ORDER BY cost_usd DESC
            """,
            year, month,
        )

    return [dict(r) for r in rows]