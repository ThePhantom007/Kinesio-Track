"""
Orchestrates initial exercise plan creation.

Responsibilities
----------------
  1. Accept an InjuryIntakeRequest + patient profile from the intake route.
  2. Optionally wait for video_intake_analyzer baseline ROM (if intake video given).
  3. Assemble IntakeContext and call claude_client.generate_initial_plan().
  4. Write the validated plan, phases, and exercises to the DB.
  5. Set PatientProfile.active_plan_id to the new plan.
  6. Return the plan_id and summary metadata to the route handler.

This service is the only place that writes ExercisePlan + PlanPhase + Exercise
rows from AI output.  All subsequent modifications go through plan_adapter.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.claude_client import ClaudeClient
from app.ai.prompt_templates.initial_plan import IntakeContext
from app.core.exceptions import PlanGenerationError
from app.core.logging import get_logger
from app.models.injury import Injury
from app.models.patient import PatientProfile
from app.models.phase import PlanPhase
from app.models.plan import ExercisePlan, PlanStatus
from app.models.exercise import Exercise
from app.schemas.intake import InjuryIntakeResponse
from app.schemas.plan import ExercisePlanAIOutput

log = get_logger(__name__)


class ExercisePlannerService:

    def __init__(self, claude_client: ClaudeClient) -> None:
        self._claude = claude_client

    async def create_plan_from_intake(
        self,
        *,
        db: AsyncSession,
        patient: PatientProfile,
        injury: Injury,
    ) -> InjuryIntakeResponse:
        """
        Full intake → plan generation pipeline.

        Args:
            db:      Async DB session (caller owns the transaction).
            patient: PatientProfile ORM instance (with user relationship loaded).
            injury:  Injury ORM instance just written by the intake route.

        Returns:
            InjuryIntakeResponse with plan_id, status, and summary metadata.

        Raises:
            PlanGenerationError: Claude failed to produce a valid plan.
        """
        log.info(
            "plan_generation_started",
            patient_id=str(patient.id),
            injury_id=str(injury.id),
            body_part=injury.body_part,
        )

        ctx = self._build_intake_context(patient, injury)

        ai_plan: ExercisePlanAIOutput = await self._claude.generate_initial_plan(
            ctx, db, patient_id=patient.id
        )

        plan = await self._persist_plan(db, patient=patient, injury=injury, ai_plan=ai_plan)

        # Point the patient at their new active plan.
        patient.active_plan_id = plan.id
        db.add(patient)

        log.info(
            "plan_generation_complete",
            plan_id=str(plan.id),
            phases=len(ai_plan.phases),
            exercises=sum(len(p.exercises) for p in ai_plan.phases),
        )

        return InjuryIntakeResponse(
            injury_id=injury.id,
            plan_id=plan.id,
            status="ready",
            estimated_phases=len(ai_plan.phases),
            estimated_weeks=ai_plan.estimated_weeks,
            summary=ai_plan.summary,
            video_processing_queued=bool(injury.intake_video_s3_key),
        )

    # ── Context assembly ───────────────────────────────────────────────────────

    def _build_intake_context(
        self,
        patient: PatientProfile,
        injury: Injury,
    ) -> IntakeContext:
        """Map ORM objects → IntakeContext dataclass for the prompt builder."""
        return IntakeContext(
            injury_description=injury.description,
            body_part=injury.body_part.value,
            pain_score=injury.pain_score,
            age=patient.age,
            activity_level=patient.activity_level.value if patient.activity_level else None,
            mobility_notes=injury.mobility_notes or patient.mobility_notes,
            baseline_rom=patient.baseline_rom,
            contraindications=[],   # populated from medical_notes by clinician post-intake
            medical_notes=patient.medical_notes,
        )

    # ── DB persistence ─────────────────────────────────────────────────────────

    async def _persist_plan(
        self,
        db: AsyncSession,
        *,
        patient: PatientProfile,
        injury: Injury,
        ai_plan: ExercisePlanAIOutput,
    ) -> ExercisePlan:
        """
        Write ExercisePlan → PlanPhase(s) → Exercise(s) in a single flush.
        No commit here — the route handler commits after success.
        """
        plan = ExercisePlan(
            patient_id=patient.id,
            injury_id=injury.id,
            title=ai_plan.title,
            version=1,
            status=PlanStatus.ACTIVE,
            current_phase=1,
            recovery_target_days=ai_plan.recovery_target_days,
            ai_generated=True,
            contraindications=ai_plan.contraindications or [],
            escalation_criteria=ai_plan.escalation_criteria or [],
        )
        db.add(plan)
        await db.flush()   # get plan.id before writing phases

        for phase_data in ai_plan.phases:
            phase = PlanPhase(
                plan_id=plan.id,
                phase_number=phase_data.phase_number,
                name=phase_data.name,
                goal=phase_data.goal,
                duration_days=phase_data.duration_days,
                progression_criteria=phase_data.progression_criteria,
            )
            db.add(phase)
            await db.flush()  # get phase.id before writing exercises

            for order_idx, ex_data in enumerate(phase_data.exercises):
                exercise = Exercise(
                    phase_id=phase.id,
                    slug=ex_data.slug,
                    name=ex_data.name,
                    order_index=order_idx,
                    sets=ex_data.sets,
                    reps=ex_data.reps,
                    hold_seconds=ex_data.hold_seconds,
                    rest_seconds=ex_data.rest_seconds,
                    tempo=ex_data.tempo,
                    target_joints=ex_data.target_joints,
                    landmark_rules={
                        joint: rule.model_dump()
                        for joint, rule in ex_data.landmark_rules.items()
                    },
                    red_flags=[rf.model_dump() for rf in ex_data.red_flags],
                    patient_instructions=ex_data.patient_instructions,
                    difficulty=ex_data.difficulty,
                )
                db.add(exercise)

        await db.flush()
        return plan