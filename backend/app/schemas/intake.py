"""
Schemas for the injury intake endpoint:
  POST /api/v1/intake

This is the entry point for the entire treatment journey.  The patient
submits a text description and optional intake video; the backend runs
video_intake_analyzer (if video supplied) and then calls Claude to generate
the initial exercise plan.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, field_validator

from app.models.injury import BodyPart
from app.schemas.base import AppBaseModel, AppResponseModel


# ── Request ───────────────────────────────────────────────────────────────────

class InjuryIntakeRequest(AppBaseModel):
    """
    Submitted by the patient (or a clinician on their behalf) at the start
    of a new treatment course.
    """

    description: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description=(
            "Free-text description of the injury, symptoms, and how it occurred. "
            "This is passed verbatim into the Claude plan-generation prompt, "
            "so richer detail produces better plans."
        ),
    )
    body_part: BodyPart = Field(
        ...,
        description="Primary anatomical location of the injury.",
    )
    pain_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported pain level at time of intake (1 = no pain, 10 = worst imaginable).",
    )
    intake_video_s3_key: str | None = Field(
        None,
        max_length=1024,
        description=(
            "S3 object key of the pre-uploaded intake video. "
            "Obtain the key by calling POST /media/upload-url first, "
            "uploading directly to S3, then supplying the key here."
        ),
    )

    @field_validator("description")
    @classmethod
    def no_pii_warning(cls, v: str) -> str:
        # Basic sanity — don't block on this, just normalise whitespace.
        return " ".join(v.split())


# ── Response ──────────────────────────────────────────────────────────────────

class InjuryIntakeResponse(AppResponseModel):
    """
    Returned synchronously after the intake POST.

    If the intake video is provided, plan generation is kicked off as a
    Celery task and the response is returned immediately with status
    ``generating``.  The client should poll GET /plans/{plan_id} or wait
    for a push notification.

    If no intake video is provided, plan generation runs inline (typically
    5–15 s) and status will be ``ready`` on the first response.
    """

    injury_id: UUID
    plan_id: UUID
    status: str = Field(
        ...,
        description="'ready' if the plan is available immediately, 'generating' if async.",
    )
    estimated_phases: int = Field(
        ...,
        description="Number of recovery phases in the generated plan.",
    )
    estimated_weeks: int = Field(
        ...,
        description="Total estimated programme duration in weeks.",
    )
    summary: str = Field(
        ...,
        description="One-paragraph plain-language summary of the plan.",
    )
    video_processing_queued: bool = Field(
        False,
        description="True when the intake video has been queued for background processing.",
    )