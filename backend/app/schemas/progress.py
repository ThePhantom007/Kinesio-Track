"""
Schemas for the progress and analytics endpoints:
  GET /api/v1/patients/{id}/progress
  GET /api/v1/patients/{id}/progress/recovery-eta
  GET /api/v1/patients/{id}/progress/report

These schemas define exactly what the frontend charting layer consumes.
Data is sourced from TimescaleDB continuous aggregates via db/queries/progress.py.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.base import AppBaseModel, AppResponseModel


# ── Query params ──────────────────────────────────────────────────────────────

class ProgressQueryParams(AppBaseModel):
    """
    Query parameters for GET /patients/{id}/progress.
    All fields are optional; sensible defaults are applied by the route handler.
    """

    from_date: date | None = Field(None, alias="from", description="Start of the date range (inclusive).")
    to_date: date | None = Field(None, alias="to", description="End of the date range (inclusive). Defaults to today.")
    joint: str | None = Field(
        None,
        description=(
            "Filter to a specific joint name, e.g. 'left_ankle'. "
            "If omitted, all joints are returned."
        ),
    )
    granularity: Literal["session", "daily", "weekly"] = Field(
        "session",
        description=(
            "session: one data point per session; "
            "daily: daily averages; "
            "weekly: weekly averages."
        ),
    )
    plan_id: UUID | None = Field(None, description="Filter to a specific plan version.")


# ── Data point types ──────────────────────────────────────────────────────────

class ROMDataPoint(AppResponseModel):
    """
    One range-of-motion measurement.
    Used for the ROM-over-time line chart.
    """

    timestamp: datetime
    session_id: UUID | None = None
    joint: str
    angle_deg: float = Field(..., description="Peak ROM angle recorded in degrees.")
    avg_angle_deg: float | None = Field(None, description="Mean angle across the session (if granularity=session).")
    baseline_angle_deg: float | None = Field(
        None,
        description="Baseline ROM for this joint from the intake video.",
    )
    improvement_pct: float | None = Field(
        None,
        description="Percentage improvement vs baseline. Positive = better.",
    )


class QualityDataPoint(AppResponseModel):
    """
    One form quality measurement.
    Used for the quality-score-over-time line chart.
    """

    timestamp: datetime
    session_id: UUID | None = None
    quality_score: float = Field(..., ge=0, le=100)
    completion_pct: float | None = Field(None, ge=0, le=1)
    post_session_pain: int | None = Field(None, ge=1, le=10)


class JointROMSeries(AppResponseModel):
    """ROM time-series for one joint."""

    joint: str
    data_points: list[ROMDataPoint]
    current_rom: float | None = Field(None, description="Most recent recorded angle.")
    target_rom: float | None = Field(None, description="Target ROM from the exercise plan.")
    baseline_rom: float | None = Field(None, description="ROM at intake baseline.")


# ── Recovery forecast ─────────────────────────────────────────────────────────

class RecoveryForecast(AppResponseModel):
    """
    AI + regression-based recovery estimate.
    Produced by recovery_forecaster service.
    """

    estimated_recovery_date: date | None = Field(
        None,
        description=(
            "Projected date of reaching the plan's target ROM. "
            "None if insufficient data (< MIN_SESSIONS_FOR_ETA sessions)."
        ),
    )
    estimated_days_remaining: int | None
    confidence: Literal["low", "moderate", "high"] = Field(
        ...,
        description=(
            "low: < 5 sessions or high variance; "
            "moderate: 5–10 sessions with reasonable trend; "
            "high: 10+ sessions with consistent improvement."
        ),
    )
    trend: Literal["improving", "plateauing", "regressing"] = Field(
        ...,
        description="Direction of quality score slope over the last N sessions.",
    )
    slope_per_session: float | None = Field(
        None,
        description="Linear regression slope of quality score per session.",
    )
    sessions_analysed: int
    ai_narrative: str | None = Field(
        None,
        description="Claude-generated plain-language explanation of the trajectory.",
    )


# ── Milestones ────────────────────────────────────────────────────────────────

class ProgressMilestone(AppResponseModel):
    """A notable achievement in the patient's recovery journey."""

    milestone_type: str = Field(
        ...,
        description=(
            "phase_complete | quality_target_reached | rom_target_reached | "
            "pain_free_session | consecutive_sessions"
        ),
    )
    label: str = Field(..., description="Human-readable description, e.g. 'Completed Phase 1'.")
    achieved_at: datetime
    value: float | None = Field(None, description="Numeric value associated with the milestone.")


# ── Full progress response ─────────────────────────────────────────────────────

class ProgressResponse(AppResponseModel):
    """
    Complete progress data payload returned by GET /patients/{id}/progress.
    Designed to be the single API call that populates the entire dashboard.
    """

    patient_id: UUID
    plan_id: UUID | None
    plan_title: str | None
    current_phase: int | None
    total_phases: int | None
    sessions_completed: int
    last_session_at: datetime | None

    # Chart series
    rom_series: list[JointROMSeries] = Field(
        default_factory=list,
        description="ROM time-series, one entry per joint.",
    )
    quality_series: list[QualityDataPoint] = Field(
        default_factory=list,
        description="Quality score time-series.",
    )

    # Summary stats
    avg_quality_score: float | None = Field(None, description="Mean quality score across all sessions in range.")
    avg_pain_score: float | None = Field(None, description="Mean post-session pain score across all sessions.")
    total_sessions_in_range: int = 0

    # Recovery projection
    recovery_forecast: RecoveryForecast | None = None

    # Milestones
    milestones: list[ProgressMilestone] = Field(default_factory=list)

    # Query metadata
    from_date: date | None
    to_date: date | None
    granularity: str
    generated_at: datetime