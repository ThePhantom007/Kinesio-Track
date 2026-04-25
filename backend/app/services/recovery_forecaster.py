"""
Computes a data-driven recovery ETA for a patient.

Algorithm
---------
  1. Fetch the last N quality_score and ROM data points from TimescaleDB.
  2. Fit a linear regression (scipy.stats.linregress) on session_number → quality_score.
  3. Project forward to the plan's target quality threshold (QUALITY_PROGRESSION_THRESHOLD).
  4. Convert sessions remaining → calendar days using the patient's session frequency.
  5. Feed the trend + slope into Claude to produce a human-readable narrative.

Confidence levels
-----------------
  high:     10+ sessions, R² > 0.6
  moderate: 5–9 sessions, R² > 0.3
  low:      < 5 sessions or high variance

Guard: if fewer than MIN_SESSIONS_FOR_ETA sessions exist, return a
RecoveryForecast with estimated_recovery_date=None and confidence="low".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import InsufficientDataError
from app.core.logging import get_logger
from app.db.queries.analytics import last_n_session_metrics, quality_trend_slope
from app.models.plan import ExercisePlan
from app.schemas.progress import RecoveryForecast

log = get_logger(__name__)

_TARGET_QUALITY = settings.QUALITY_PROGRESSION_THRESHOLD  # default 78.0


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    sessions_to_target: int | None
    trend: str  # "improving" | "plateauing" | "regressing"


class RecoveryForecasterService:

    async def forecast(
        self,
        *,
        db: AsyncSession,
        patient_id: UUID,
        plan: ExercisePlan,
    ) -> RecoveryForecast:
        """
        Compute the recovery forecast for *patient_id* on *plan*.

        Args:
            db:         Async DB session.
            patient_id: UUID of the patient.
            plan:       Active ExercisePlan ORM object.

        Returns:
            RecoveryForecast schema instance — always returns, never raises.
            If insufficient data, confidence="low" and no date is set.
        """
        metrics = await last_n_session_metrics(patient_id, plan.id, n=20)

        if len(metrics) < settings.MIN_SESSIONS_FOR_ETA:
            log.info(
                "forecast_insufficient_data",
                patient_id=str(patient_id),
                sessions=len(metrics),
                required=settings.MIN_SESSIONS_FOR_ETA,
            )
            return RecoveryForecast(
                estimated_recovery_date=None,
                estimated_days_remaining=None,
                confidence="low",
                trend="improving",
                slope_per_session=None,
                sessions_analysed=len(metrics),
                ai_narrative=None,
            )

        regression = self._fit_regression(metrics)
        confidence  = self._confidence_level(len(metrics), regression.r_squared)
        recovery_date, days_remaining = self._project_date(
            regression, metrics, plan
        )

        log.info(
            "forecast_computed",
            patient_id=str(patient_id),
            trend=regression.trend,
            slope=round(regression.slope, 3),
            r_squared=round(regression.r_squared, 3),
            confidence=confidence,
            days_remaining=days_remaining,
        )

        return RecoveryForecast(
            estimated_recovery_date=recovery_date,
            estimated_days_remaining=days_remaining,
            confidence=confidence,
            trend=regression.trend,
            slope_per_session=round(regression.slope, 3),
            sessions_analysed=len(metrics),
            ai_narrative=None,   # populated by the analytics Celery task if needed
        )

    # ── Regression ─────────────────────────────────────────────────────────────

    def _fit_regression(self, metrics: list[dict[str, Any]]) -> RegressionResult:
        """Fit a linear regression on session_index → avg_quality_score."""
        xs = list(range(len(metrics)))
        ys = [
            m.get("avg_quality_score") or 0.0
            for m in metrics
        ]

        if len(set(ys)) == 1:
            # Flat line — no variance to regress
            return RegressionResult(
                slope=0.0,
                intercept=ys[0],
                r_squared=0.0,
                sessions_to_target=None,
                trend="plateauing",
            )

        result = stats.linregress(xs, ys)
        slope     = float(result.slope)
        intercept = float(result.intercept)
        r_squared = float(result.rvalue ** 2)

        # Sessions needed to reach target from current level
        current = ys[-1]
        sessions_to_target: int | None = None
        if slope > 0.1 and current < _TARGET_QUALITY:
            raw = (_TARGET_QUALITY - current) / slope
            sessions_to_target = max(1, int(round(raw)))
        elif current >= _TARGET_QUALITY:
            sessions_to_target = 0

        trend = (
            "improving"   if slope >  0.5 else
            "regressing"  if slope < -0.5 else
            "plateauing"
        )

        return RegressionResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            sessions_to_target=sessions_to_target,
            trend=trend,
        )

    def _project_date(
        self,
        regression: RegressionResult,
        metrics: list[dict[str, Any]],
        plan: ExercisePlan,
    ) -> tuple[date | None, int | None]:
        """
        Convert sessions_to_target → calendar date using historical session frequency.
        """
        if regression.sessions_to_target is None:
            return None, None
        if regression.sessions_to_target == 0:
            return date.today(), 0

        # Estimate sessions per week from the last few metrics
        freq_per_week = self._estimate_session_frequency(metrics)
        if freq_per_week <= 0:
            freq_per_week = 3.0   # safe default

        days_remaining = int(
            (regression.sessions_to_target / freq_per_week) * 7
        )
        recovery_date = date.today() + timedelta(days=days_remaining)
        return recovery_date, days_remaining

    def _estimate_session_frequency(self, metrics: list[dict[str, Any]]) -> float:
        """
        Estimate sessions per week from the last 8 metric timestamps.
        Returns a float (e.g. 3.0 = 3 sessions/week).
        """
        dates = []
        for m in metrics[-8:]:
            d = m.get("session_date")
            if d:
                if isinstance(d, str):
                    from datetime import datetime
                    d = datetime.fromisoformat(d).date()
                dates.append(d)

        if len(dates) < 2:
            return 3.0

        dates_sorted = sorted(dates)
        span_days = (dates_sorted[-1] - dates_sorted[0]).days
        if span_days == 0:
            return 3.0

        sessions_per_day = (len(dates) - 1) / span_days
        return sessions_per_day * 7

    def _confidence_level(self, n_sessions: int, r_squared: float) -> str:
        if n_sessions >= 10 and r_squared >= 0.6:
            return "high"
        if n_sessions >= 5 and r_squared >= 0.3:
            return "moderate"
        return "low"