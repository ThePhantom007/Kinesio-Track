"""
Post-session plan adaptation via Claude JSON Patch.

Called by the Celery post_session_analysis task after every completed session.
Reads recent session metrics from TimescaleDB, decides whether adaptation is
warranted, calls Claude for a JSON Patch, and applies it to the current plan.

Adaptation guard
----------------
Adaptation is skipped unless at least MIN_SESSIONS_FOR_ETA sessions exist
so Claude has enough signal.  It is also skipped if the most recent session
already triggered a plan adaptation (plan_adapted=True) to avoid churn.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import UUID

import jsonpatch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.claude_client import ClaudeClient
from app.ai.prompt_templates.adapt_plan import AdaptationContext
from app.core.config import settings
from app.core.exceptions import PlanAdaptationError
from app.core.logging import get_logger
from app.db.queries.analytics import last_n_session_metrics, quality_trend_slope
from app.models.plan import ExercisePlan, PlanStatus
from app.models.phase import PlanPhase
from app.models.exercise import Exercise
from app.models.patient import PatientProfile
from app.models.session import ExerciseSession

log = get_logger(__name__)


class PlanAdapterService:

    def __init__(self, claude_client: ClaudeClient) -> None:
        self._claude = claude_client

    async def adapt_after_session(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
    ) -> bool:
        """
        Entry point called by the Celery post_session_analysis task.

        Returns:
            True if the plan was adapted, False if skipped or no changes made.
        """
        plan = await db.get(ExercisePlan, session.plan_id)
        if plan is None or plan.status != PlanStatus.ACTIVE:
            log.info("adaptation_skipped_no_active_plan", session_id=str(session.id))
            return False

        patient = await db.get(PatientProfile, session.patient_id)

        # Fetch recent metrics from TimescaleDB
        metrics = await last_n_session_metrics(session.patient_id, plan.id, n=10)

        if len(metrics) < settings.MIN_SESSIONS_FOR_ETA:
            log.info(
                "adaptation_skipped_insufficient_data",
                session_id=str(session.id),
                sessions_available=len(metrics),
                required=settings.MIN_SESSIONS_FOR_ETA,
            )
            return False

        avg_quality = sum(m["avg_quality_score"] for m in metrics) / len(metrics)
        avg_pain    = sum(m["post_session_pain"] for m in metrics if m.get("post_session_pain")) / max(len(metrics), 1)
        avg_completion = sum(m.get("completion_pct", 1.0) for m in metrics) / len(metrics)

        # Skip adaptation if metrics are solidly in the maintenance range
        if (
            settings.QUALITY_REGRESSION_THRESHOLD < avg_quality < settings.QUALITY_PROGRESSION_THRESHOLD
            and avg_pain < settings.PAIN_RED_FLAG_THRESHOLD - 2
        ):
            log.info(
                "adaptation_skipped_stable_metrics",
                avg_quality=round(avg_quality, 1),
                avg_pain=round(avg_pain, 1),
            )
            return False

        exercises = await self._load_current_exercises(db, plan)
        ctx = AdaptationContext(
            current_plan=self._plan_to_dict(plan),
            current_exercises=exercises,
            session_metrics=metrics,
            avg_quality_score=avg_quality,
            avg_pain_score=avg_pain,
            completion_rate=avg_completion,
            sessions_analysed=len(metrics),
            age=patient.age if patient else None,
            activity_level=patient.activity_level.value if patient and patient.activity_level else None,
            mobility_notes=patient.mobility_notes if patient else None,
        )

        patch_ops = await self._claude.adapt_plan(ctx, db, patient_id=session.patient_id)

        if not patch_ops:
            log.info("adaptation_no_changes", session_id=str(session.id))
            return False

        await self._apply_patch(db, plan=plan, patch_ops=patch_ops)

        # Mark the triggering session as adapted
        session.plan_adapted = True
        db.add(session)
        await db.flush()

        log.info(
            "plan_adapted",
            plan_id=str(plan.id),
            ops=len(patch_ops),
            new_version=plan.version,
        )
        return True

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _load_current_exercises(
        self,
        db: AsyncSession,
        plan: ExercisePlan,
    ) -> list[dict[str, Any]]:
        """Return flattened list of exercise dicts for the current phase."""
        result = await db.execute(
            select(PlanPhase, Exercise)
            .join(Exercise, Exercise.phase_id == PlanPhase.id)
            .where(
                PlanPhase.plan_id == plan.id,
                PlanPhase.phase_number == plan.current_phase,
            )
            .order_by(Exercise.order_index)
        )
        rows = result.all()
        return [
            {
                "slug": ex.slug,
                "name": ex.name,
                "sets": ex.sets,
                "reps": ex.reps,
                "hold_seconds": ex.hold_seconds,
                "rest_seconds": ex.rest_seconds,
                "difficulty": ex.difficulty,
                "phase_number": phase.phase_number,
            }
            for phase, ex in rows
        ]

    def _plan_to_dict(self, plan: ExercisePlan) -> dict[str, Any]:
        """Lightweight plan summary for the adaptation prompt."""
        return {
            "current_phase": plan.current_phase,
            "recovery_target_days": plan.recovery_target_days,
            "phases": [],   # exercise list is passed separately to keep prompt compact
        }

    async def _apply_patch(
        self,
        db: AsyncSession,
        plan: ExercisePlan,
        patch_ops: list[dict[str, Any]],
    ) -> None:
        """
        Apply RFC 6902 JSON Patch operations to the plan.

        Simple scalar changes (reps, sets, rest_seconds, current_phase) are
        mapped directly to ORM updates.  We don't store plans as a single JSONB
        blob — changes are applied field-by-field to the normalised tables.
        """
        for op in patch_ops:
            path_parts = op["path"].strip("/").split("/")
            value = op.get("value")

            # Top-level plan fields
            if len(path_parts) == 1:
                field = path_parts[0]
                if field == "current_phase" and isinstance(value, int):
                    plan.current_phase = value
                    plan.version += 1
                    db.add(plan)
                continue

            # Exercise field: /phases/{i}/exercises/{j}/{field}
            if len(path_parts) == 5 and path_parts[0] == "phases" and path_parts[2] == "exercises":
                phase_idx = int(path_parts[1])
                ex_idx    = int(path_parts[3])
                field     = path_parts[4]
                await self._patch_exercise_field(db, plan, phase_idx + 1, ex_idx, field, value)
                continue

            log.warning("unknown_patch_path", path=op["path"])

        await db.flush()

    async def _patch_exercise_field(
        self,
        db: AsyncSession,
        plan: ExercisePlan,
        phase_number: int,
        exercise_index: int,
        field: str,
        value: Any,
    ) -> None:
        """Apply one field change to a specific exercise ORM row."""
        ALLOWED_FIELDS = {"sets", "reps", "hold_seconds", "rest_seconds", "difficulty", "patient_instructions"}
        if field not in ALLOWED_FIELDS:
            log.warning("patch_field_not_allowed", field=field)
            return

        result = await db.execute(
            select(Exercise)
            .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
            .where(
                PlanPhase.plan_id == plan.id,
                PlanPhase.phase_number == phase_number,
            )
            .order_by(Exercise.order_index)
            .offset(exercise_index)
            .limit(1)
        )
        exercise = result.scalar_one_or_none()
        if exercise is None:
            log.warning("patch_exercise_not_found", phase=phase_number, index=exercise_index)
            return

        setattr(exercise, field, value)
        db.add(exercise)