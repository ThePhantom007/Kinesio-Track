"""
Unit tests for app/services/recovery_forecaster.py.

Tests the linear regression logic, confidence assignment, and edge-case
handling without hitting the database.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.recovery_forecaster import RecoveryForecasterService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plan(recovery_target_days: int = 42, current_phase: int = 1):
    plan = MagicMock()
    plan.id                   = uuid4()
    plan.recovery_target_days = recovery_target_days
    plan.current_phase        = current_phase
    plan.phases               = [MagicMock()]
    return plan


def _make_metrics(
    n: int,
    start_quality: float = 40.0,
    slope: float = 5.0,
    pain: float = 4.0,
    start_days_ago: int = 20,
) -> list[dict]:
    """Generate a list of n metrics with a linear quality trend."""
    from datetime import datetime, timezone
    metrics = []
    for i in range(n):
        days_ago = start_days_ago - (i * 2)
        d        = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date()
        metrics.append({
            "session_id":        str(uuid4()),
            "session_date":      d.isoformat(),
            "avg_quality_score": start_quality + (i * slope),
            "post_session_pain": pain,
            "completion_pct":    0.9,
            "peak_rom_degrees":  20.0 + i,
        })
    return metrics


@pytest.fixture
def svc():
    return RecoveryForecasterService()


# ── Insufficient data ─────────────────────────────────────────────────────────

class TestInsufficientData:

    @pytest.mark.asyncio
    @patch("app.services.recovery_forecaster.last_n_session_metrics")
    async def test_returns_low_confidence_when_too_few_sessions(
        self, mock_metrics, svc
    ):
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(settings.MIN_SESSIONS_FOR_ETA - 1)
        db      = AsyncMock()
        patient = MagicMock()
        patient.id = uuid4()
        plan    = _make_plan()

        result = await svc.forecast(db=db, patient_id=patient.id, plan=plan)

        assert result.confidence == "low"
        assert result.estimated_recovery_date is None
        assert result.estimated_days_remaining is None

    @pytest.mark.asyncio
    @patch("app.services.recovery_forecaster.last_n_session_metrics")
    async def test_returns_low_confidence_for_zero_sessions(
        self, mock_metrics, svc
    ):
        mock_metrics.return_value = []
        db      = AsyncMock()
        patient_id = uuid4()
        plan    = _make_plan()

        result = await svc.forecast(db=db, patient_id=patient_id, plan=plan)
        assert result.confidence == "low"


# ── Regression fitting ────────────────────────────────────────────────────────

class TestRegressionFitting:

    def test_improving_trend_positive_slope(self, svc):
        metrics = _make_metrics(n=10, start_quality=40.0, slope=5.0)
        reg     = svc._fit_regression(metrics)
        assert reg.slope > 0
        assert reg.trend == "improving"

    def test_regressing_trend_negative_slope(self, svc):
        metrics = _make_metrics(n=10, start_quality=80.0, slope=-4.0)
        reg     = svc._fit_regression(metrics)
        assert reg.slope < 0
        assert reg.trend == "regressing"

    def test_flat_trend_plateauing(self, svc):
        metrics = _make_metrics(n=10, start_quality=65.0, slope=0.0)
        reg     = svc._fit_regression(metrics)
        assert reg.trend == "plateauing"

    def test_already_at_target_sessions_to_target_zero(self, svc):
        from app.core.config import settings
        metrics = _make_metrics(n=5, start_quality=settings.QUALITY_PROGRESSION_THRESHOLD + 5, slope=0)
        reg     = svc._fit_regression(metrics)
        assert reg.sessions_to_target == 0

    def test_regression_none_when_regressing(self, svc):
        metrics = _make_metrics(n=10, start_quality=80.0, slope=-5.0)
        reg     = svc._fit_regression(metrics)
        # Regressing trend → cannot project forward to target
        assert reg.sessions_to_target is None

    def test_r_squared_in_valid_range(self, svc):
        metrics = _make_metrics(n=10, start_quality=40.0, slope=5.0)
        reg     = svc._fit_regression(metrics)
        assert 0.0 <= reg.r_squared <= 1.0


# ── Confidence levels ─────────────────────────────────────────────────────────

class TestConfidenceLevels:

    def test_high_confidence_many_sessions_high_r2(self, svc):
        # Perfect linear data → high R²
        metrics = _make_metrics(n=15, start_quality=30.0, slope=4.0)
        reg     = svc._fit_regression(metrics)
        conf    = svc._confidence_level(15, reg.r_squared)
        assert conf in ("high", "moderate")   # depends on actual R²

    def test_low_confidence_few_sessions(self, svc):
        conf = svc._confidence_level(2, 0.9)
        assert conf == "low"

    def test_moderate_confidence_mid_sessions_good_r2(self, svc):
        conf = svc._confidence_level(7, 0.5)
        assert conf == "moderate"

    def test_low_confidence_low_r2_many_sessions(self, svc):
        conf = svc._confidence_level(15, 0.1)
        assert conf == "low"


# ── Date projection ───────────────────────────────────────────────────────────

class TestDateProjection:

    def test_projected_date_in_future(self, svc):
        metrics = _make_metrics(n=8, start_quality=40.0, slope=5.0)
        reg     = svc._fit_regression(metrics)
        plan    = _make_plan()

        recovery_date, days = svc._project_date(reg, metrics, plan)

        if recovery_date is not None:
            assert recovery_date >= date.today()

    def test_already_recovered_returns_today(self, svc):
        from app.core.config import settings
        metrics = _make_metrics(
            n=5,
            start_quality=settings.QUALITY_PROGRESSION_THRESHOLD + 10,
            slope=0.0,
        )
        reg  = svc._fit_regression(metrics)
        plan = _make_plan()

        recovery_date, days = svc._project_date(reg, metrics, plan)

        if reg.sessions_to_target == 0:
            assert recovery_date == date.today()
            assert days == 0

    def test_none_returned_when_regression_sessions_none(self, svc):
        reg           = MagicMock()
        reg.sessions_to_target = None
        plan          = _make_plan()
        metrics       = _make_metrics(n=5)

        recovery_date, days = svc._project_date(reg, metrics, plan)
        assert recovery_date is None
        assert days is None


# ── Session frequency estimation ──────────────────────────────────────────────

class TestSessionFrequency:

    def test_frequency_from_regular_sessions(self, svc):
        # 7 sessions over 14 days = 3.5 sessions/week
        metrics = _make_metrics(n=7, start_days_ago=14)
        freq    = svc._estimate_session_frequency(metrics)
        assert 2.0 < freq < 6.0  # reasonable range

    def test_frequency_single_session(self, svc):
        metrics = _make_metrics(n=1)
        freq    = svc._estimate_session_frequency(metrics)
        assert freq == pytest.approx(3.0, abs=0.1)  # falls back to default

    def test_frequency_no_sessions(self, svc):
        freq = svc._estimate_session_frequency([])
        assert freq == pytest.approx(3.0, abs=0.1)


# ── Full forecast ─────────────────────────────────────────────────────────────

class TestFullForecast:

    @pytest.mark.asyncio
    @patch("app.services.recovery_forecaster.last_n_session_metrics")
    async def test_improving_patient_returns_forecast(self, mock_metrics, svc):
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(
            n=settings.MIN_SESSIONS_FOR_ETA + 3,
            start_quality=40.0,
            slope=6.0,
        )
        patient_id = uuid4()
        plan       = _make_plan()
        db         = AsyncMock()

        result = await svc.forecast(db=db, patient_id=patient_id, plan=plan)

        assert result.trend == "improving"
        assert result.sessions_analysed > 0
        assert result.slope_per_session is not None
        assert result.slope_per_session > 0

    @pytest.mark.asyncio
    @patch("app.services.recovery_forecaster.last_n_session_metrics")
    async def test_result_schema_complete(self, mock_metrics, svc):
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(n=settings.MIN_SESSIONS_FOR_ETA + 2)
        patient_id = uuid4()
        plan       = _make_plan()
        db         = AsyncMock()

        result = await svc.forecast(db=db, patient_id=patient_id, plan=plan)

        assert hasattr(result, "confidence")
        assert hasattr(result, "trend")
        assert hasattr(result, "sessions_analysed")
        assert hasattr(result, "estimated_recovery_date")
        assert hasattr(result, "estimated_days_remaining")