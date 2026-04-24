"""
Schemas for exercise plan endpoints:
  GET   /api/v1/plans/{id}
  GET   /api/v1/plans/{id}/exercises
  PATCH /api/v1/plans/{id}              (clinician override)

These schemas are also the contract for what Claude must return — the
response_parser in app/ai/response_parser.py validates Claude's JSON output
against ExercisePlanAIOutput before it ever touches the DB.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.models.plan import PlanStatus
from app.schemas.base import AppBaseModel, AppResponseModel


# ── Landmark rules (shared with AI layer) ────────────────────────────────────

class JointRule(AppBaseModel):
    """
    Acceptable angle range for one joint during an exercise.
    Consumed by pose_analyzer on every frame.
    """

    min_angle: float = Field(..., ge=0, le=360, description="Minimum acceptable joint angle in degrees.")
    max_angle: float = Field(..., ge=0, le=360, description="Maximum acceptable joint angle in degrees.")
    axis: str = Field(
        "sagittal",
        description="Movement plane: sagittal | frontal | transverse.",
    )
    priority: str = Field(
        "primary",
        description=(
            "primary: always checked; secondary: checked only if primary joints are within range; "
            "bilateral: triggers bilateral_asymmetry check instead of absolute angle check."
        ),
    )

    @model_validator(mode="after")
    def min_less_than_max(self) -> "JointRule":
        if self.min_angle >= self.max_angle:
            raise ValueError(
                f"min_angle ({self.min_angle}) must be less than max_angle ({self.max_angle})."
            )
        return self


class RedFlagRule(AppBaseModel):
    """One condition that triggers immediate red_flag_monitor escalation."""

    condition: str = Field(..., description="Expression evaluated against current frame metrics, e.g. 'left_knee.angle < 40'.")
    action: str = Field(..., description="stop | warn | seek_care")
    reason: str = Field(..., description="Plain-language reason shown in the red flag record.")


# ── Exercise ──────────────────────────────────────────────────────────────────

class ExerciseResponse(AppResponseModel):
    """Full exercise detail including pose analysis rules."""

    id: UUID
    phase_id: UUID
    slug: str
    name: str
    order_index: int
    sets: int
    reps: int
    hold_seconds: int
    rest_seconds: int
    tempo: str | None
    target_joints: list[str]
    landmark_rules: dict[str, JointRule] = Field(
        ...,
        description="Per-joint acceptable angle ranges. Keys are MediaPipe joint names.",
    )
    red_flags: list[RedFlagRule] | None = None
    patient_instructions: str | None
    reference_video_url: str | None
    difficulty: str | None
    created_at: datetime


class ExerciseSummary(AppResponseModel):
    """Condensed exercise view for plan list endpoints."""

    id: UUID
    slug: str
    name: str
    order_index: int
    sets: int
    reps: int
    hold_seconds: int
    difficulty: str | None
    target_joints: list[str]


# ── Phase ─────────────────────────────────────────────────────────────────────

class PlanPhaseResponse(AppResponseModel):
    id: UUID
    plan_id: UUID
    phase_number: int
    name: str
    goal: str
    duration_days: int
    progression_criteria: str | None
    exercises: list[ExerciseResponse]
    created_at: datetime


class PlanPhaseSummary(AppResponseModel):
    """Phase view without full exercise details — for plan overview endpoints."""

    id: UUID
    phase_number: int
    name: str
    goal: str
    duration_days: int
    exercise_count: int


# ── Plan ──────────────────────────────────────────────────────────────────────

class ExercisePlanResponse(AppResponseModel):
    """Full plan including all phases and exercises."""

    id: UUID
    patient_id: UUID
    injury_id: UUID
    title: str
    version: int
    status: PlanStatus
    current_phase: int
    recovery_target_days: int | None
    ai_generated: bool
    contraindications: list[str] | None
    clinician_notes: str | None
    phases: list[PlanPhaseResponse]
    created_at: datetime
    updated_at: datetime


class ExercisePlanSummary(AppResponseModel):
    """Plan summary without phase/exercise detail."""

    id: UUID
    title: str
    version: int
    status: PlanStatus
    current_phase: int
    total_phases: int
    recovery_target_days: int | None
    created_at: datetime
    updated_at: datetime


# ── Clinician override ────────────────────────────────────────────────────────

class ExercisePatchRequest(AppBaseModel):
    """Fields a clinician can modify on an individual exercise."""

    sets: int | None = Field(None, ge=1, le=20)
    reps: int | None = Field(None, ge=1, le=100)
    hold_seconds: int | None = Field(None, ge=0, le=300)
    rest_seconds: int | None = Field(None, ge=0, le=600)
    tempo: str | None = None
    patient_instructions: str | None = Field(None, max_length=4000)
    landmark_rules: dict[str, JointRule] | None = None
    red_flags: list[RedFlagRule] | None = None
    difficulty: str | None = None


class PlanPatchRequest(AppBaseModel):
    """
    Clinician override applied to an existing plan.
    Creates a new plan version with ai_generated=False.
    Supply only the fields you want to change.
    """

    title: str | None = Field(None, min_length=1, max_length=256)
    contraindications: list[str] | None = None
    clinician_notes: str | None = Field(None, max_length=8000)
    recovery_target_days: int | None = Field(None, ge=1, le=730)
    current_phase: int | None = Field(None, ge=1)
    exercise_patches: dict[str, ExercisePatchRequest] | None = Field(
        None,
        description="Map of exercise_id (string UUID) → fields to change.",
    )


# ── AI output schema (used by response_parser) ───────────────────────────────

class ExerciseAIOutput(AppBaseModel):
    """
    Shape Claude must output for each exercise.
    Validated by response_parser before DB writes.
    """

    slug: str = Field(..., pattern=r"^[a-z0-9-]+$")
    name: str
    sets: int = Field(..., ge=1)
    reps: int = Field(..., ge=1)
    hold_seconds: int = Field(0, ge=0)
    rest_seconds: int = Field(30, ge=0)
    tempo: str | None = None
    target_joints: list[str] = Field(..., min_length=1)
    landmark_rules: dict[str, JointRule]
    red_flags: list[RedFlagRule] = Field(default_factory=list)
    patient_instructions: str
    difficulty: str = "beginner"
    safety_warnings: list[str] = Field(default_factory=list)

    @field_validator("target_joints")
    @classmethod
    def valid_joint_names(cls, v: list[str]) -> list[str]:
        valid = {
            "left_ankle", "right_ankle", "left_knee", "right_knee",
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "neck", "lumbar_spine", "thoracic_spine",
        }
        for joint in v:
            if joint not in valid:
                raise ValueError(f"Unknown joint name: '{joint}'. Must be one of {sorted(valid)}.")
        return v


class PlanPhaseAIOutput(AppBaseModel):
    phase_number: int = Field(..., ge=1)
    name: str
    goal: str
    duration_days: int = Field(..., ge=1)
    progression_criteria: str | None = None
    exercises: list[ExerciseAIOutput] = Field(..., min_length=1)


class ExercisePlanAIOutput(AppBaseModel):
    """
    Root schema that Claude must return for plan generation.
    Validated by response_parser.validate_initial_plan().
    """

    title: str
    summary: str
    estimated_weeks: int = Field(..., ge=1, le=52)
    recovery_target_days: int = Field(..., ge=7)
    contraindications: list[str] = Field(default_factory=list)
    escalation_criteria: list[dict[str, Any]] = Field(default_factory=list)
    phases: list[PlanPhaseAIOutput] = Field(..., min_length=1)

    @field_validator("phases")
    @classmethod
    def phases_sequential(cls, v: list[PlanPhaseAIOutput]) -> list[PlanPhaseAIOutput]:
        for i, phase in enumerate(v, 1):
            if phase.phase_number != i:
                raise ValueError(
                    f"Phase {i} has phase_number={phase.phase_number}. Phases must be sequential starting from 1."
                )
        return v