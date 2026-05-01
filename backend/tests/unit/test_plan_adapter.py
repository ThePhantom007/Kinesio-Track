"""
Unit tests for app/services/plan_adapter.py.

Tests the adaptation decision logic and JSON Patch application in isolation.
Claude is mocked — these tests verify the service orchestration, not the AI.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from app.services.plan_adapter import PlanAdapterService
from app.models.plan import PlanStatus
from tests.fixtures.mock_claude_responses import (
    VALID_PATCH_RESPONSE,
    EMPTY_PATCH_RESPONSE,
    PROGRESSION_PATCH_RESPONSE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(
    *,
    quality: float = 72.0,
    pain: int = 4,
    completion: float = 0.9,
    plan_adapted: bool = False,
):
    session = MagicMock()
    session.id              = uuid4()
    session.patient_id      = uuid4()
    session.plan_id         = uuid4()
    session.exercise_id     = uuid4()
    session.avg_quality_score = quality
    session.post_session_pain = pain
    session.completion_pct    = completion
    session.plan_adapted      = plan_adapted
    return session


def _make_plan(*, status=PlanStatus.ACTIVE, current_phase: int = 1):
    plan = MagicMock()
    plan.id             = uuid4()
    plan.status         = status
    plan.current_phase  = current_phase
    plan.recovery_target_days = 42
    plan.phases         = []
    return plan


def _make_metrics(
    n: int = 5,
    avg_quality: float = 72.0,
    avg_pain: float = 4.0,
    avg_completion: float = 0.9,
) -> list[dict]:
    return [
        {
            "session_id":       str(uuid4()),
            "session_date":     f"2026-04-{i+1:02d}",
            "avg_quality_score": avg_quality + (i * 0.5),
            "post_session_pain": avg_pain,
            "completion_pct":    avg_completion,
            "peak_rom_degrees":  22.0 + i,
        }
        for i in range(n)
    ]


# ── Adaptation skipping logic ─────────────────────────────────────────────────

class TestAdaptationGuards:

    @pytest_asyncio.fixture
    def svc(self):
        return PlanAdapterService(AsyncMock())

    @pytest.mark.asyncio
    async def test_skips_when_no_active_plan(self, svc):
        db      = AsyncMock()
        db.get  = AsyncMock(return_value=None)
        session = _make_session()

        result = await svc.adapt_after_session(db=db, session=session)
        assert result is False

    @pytest.mark.asyncio
    async def test_skips_when_plan_not_active(self, svc):
        plan         = _make_plan(status=PlanStatus.COMPLETED)
        db           = AsyncMock()
        db.get       = AsyncMock(return_value=plan)
        session      = _make_session()

        result = await svc.adapt_after_session(db=db, session=session)
        assert result is False

    @pytest.mark.asyncio
    @patch("app.services.plan_adapter.last_n_session_metrics")
    async def test_skips_when_insufficient_sessions(self, mock_metrics, svc):
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(n=settings.MIN_SESSIONS_FOR_ETA - 1)

        plan    = _make_plan()
        db      = AsyncMock()
        db.get  = AsyncMock(return_value=plan)
        session = _make_session()

        result = await svc.adapt_after_session(db=db, session=session)
        assert result is False

    @pytest.mark.asyncio
    @patch("app.services.plan_adapter.last_n_session_metrics")
    async def test_skips_stable_metrics(self, mock_metrics, svc):
        """Metrics in stable range should not trigger adaptation."""
        from app.core.config import settings
        # Quality between regression and progression thresholds, low pain
        mid_quality = (
            settings.QUALITY_REGRESSION_THRESHOLD
            + settings.QUALITY_PROGRESSION_THRESHOLD
        ) / 2
        mock_metrics.return_value = _make_metrics(
            n=settings.MIN_SESSIONS_FOR_ETA + 1,
            avg_quality=mid_quality,
            avg_pain=3.0,
        )

        plan    = _make_plan()
        db      = AsyncMock()
        db.get  = AsyncMock(return_value=plan)
        session = _make_session(quality=mid_quality, pain=3)

        result = await svc.adapt_after_session(db=db, session=session)
        assert result is False


# ── Adaptation execution ──────────────────────────────────────────────────────

class TestAdaptationExecution:

    @pytest.mark.asyncio
    @patch("app.services.plan_adapter.last_n_session_metrics")
    @patch("app.services.plan_adapter.quality_trend_slope")
    async def test_adaptation_calls_claude_and_returns_true(
        self, mock_slope, mock_metrics
    ):
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(
            n=settings.MIN_SESSIONS_FOR_ETA + 1,
            avg_quality=35.0,   # below regression threshold → triggers
            avg_pain=8.0,       # high pain → triggers
        )
        mock_slope.return_value = {"slope": -1.5, "trend": "regressing"}

        claude = AsyncMock()
        claude.adapt_plan = AsyncMock(return_value=VALID_PATCH_RESPONSE)
        svc    = PlanAdapterService(claude)

        plan = _make_plan()

        db      = AsyncMock()
        db.get  = AsyncMock(return_value=plan)
        db.execute = AsyncMock(return_value=MagicMock(all=lambda: []))
        db.flush   = AsyncMock()
        db.add     = MagicMock()

        session = _make_session(quality=35.0, pain=8)

        with patch.object(svc, "_load_current_exercises", new_callable=AsyncMock, return_value=[]):
            with patch.object(svc, "_apply_patch", new_callable=AsyncMock):
                result = await svc.adapt_after_session(db=db, session=session)

        # Claude was called
        assert claude.adapt_plan.called

    @pytest.mark.asyncio
    @patch("app.services.plan_adapter.last_n_session_metrics")
    async def test_empty_patch_returns_false(self, mock_metrics):
        """Claude returning [] means no adaptation needed."""
        from app.core.config import settings
        mock_metrics.return_value = _make_metrics(
            n=settings.MIN_SESSIONS_FOR_ETA + 1,
            avg_quality=30.0,
            avg_pain=9.0,
        )

        claude = AsyncMock()
        claude.adapt_plan = AsyncMock(return_value=EMPTY_PATCH_RESPONSE)
        svc    = PlanAdapterService(claude)

        plan    = _make_plan()
        db      = AsyncMock()
        db.get  = AsyncMock(return_value=plan)
        session = _make_session(quality=30.0, pain=9)

        with patch.object(svc, "_load_current_exercises", new_callable=AsyncMock, return_value=[]):
            result = await svc.adapt_after_session(db=db, session=session)

        assert result is False


# ── _plan_to_dict ─────────────────────────────────────────────────────────────

class TestPlanToDict:

    def test_returns_dict_with_expected_keys(self):
        svc  = PlanAdapterService(AsyncMock())
        plan = _make_plan(current_phase=2)
        d    = svc._plan_to_dict(plan)
        assert d["current_phase"] == 2
        assert "phases" in d

    def test_does_not_raise_on_none_fields(self):
        svc  = PlanAdapterService(AsyncMock())
        plan = _make_plan()
        plan.recovery_target_days = None
        d    = svc._plan_to_dict(plan)
        assert isinstance(d, dict)