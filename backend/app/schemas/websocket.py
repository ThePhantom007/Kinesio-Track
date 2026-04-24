"""
WebSocket message schemas for the real-time session feedback channel:
  WS /ws/session/{session_id}

Message flow
------------
  Client → Server:  LandmarkFrame        (pose data every 3–5 frames)
  Client → Server:  RepComplete          (client signals a rep is done)
  Client → Server:  PingMessage          (keep-alive)

  Server → Client:  FeedbackMessage      (form correction)
  Server → Client:  MilestoneMessage     (rep/set completed)
  Server → Client:  ExerciseDoneMessage  (advance to next exercise)
  Server → Client:  SessionSummaryMessage (session complete)
  Server → Client:  RedFlagMessage       (stop / seek care)
  Server → Client:  PongMessage          (keep-alive reply)
  Server → Client:  ErrorMessage         (recoverable protocol error)

Design notes
------------
- All messages carry a ``type`` discriminator field so the client can
  switch on it without inspecting other fields.
- Inbound Landmark coordinates are normalised 0.0–1.0 (MediaPipe default).
- The server sends FeedbackMessage at most once per POSE_VIOLATION_FRAME_COUNT
  frames for the same joint to avoid message flooding.
- Android (MediaPipe Tasks on-device): sends LandmarkFrame with
  source="mediapipe_tasks" — landmarks are already extracted.
- Web browser: sends LandmarkFrame with source="raw_frame" — server runs
  MediaPipe server-side via mediapipe/pose_estimator.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import Field

from app.schemas.base import AppBaseModel, AppResponseModel


# ═════════════════════════════════════════════════════════════════════════════
# INBOUND  (Client → Server)
# ═════════════════════════════════════════════════════════════════════════════

class Landmark(AppBaseModel):
    """
    One of the 33 MediaPipe Pose keypoints.
    Coordinates are normalised to [0.0, 1.0] relative to the frame dimensions.
    """

    id: int = Field(..., ge=0, le=32, description="MediaPipe landmark index (0–32).")
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    z: float = Field(..., description="Depth estimate relative to the hip midpoint. Not normalised.")
    visibility: float = Field(..., ge=0.0, le=1.0, description="Model confidence that the landmark is visible.")


class LandmarkFrame(AppBaseModel):
    """
    Sent by the client on every analysed frame.

    Android (MediaPipe Tasks):
        source = "mediapipe_tasks"
        landmarks = full 33-point list extracted on-device

    Web browser:
        source = "raw_frame"
        raw_frame_b64 = base64-encoded JPEG
        landmarks = [] (server extracts them)
    """

    type: Literal["FRAME_DATA"] = "FRAME_DATA"
    session_id: UUID
    exercise_id: UUID
    timestamp_ms: int = Field(..., description="Client-side monotonic timestamp in milliseconds.")
    frame_index: int = Field(..., ge=0, description="Zero-based frame counter within the session.")
    source: Literal["mediapipe_tasks", "raw_frame"] = "mediapipe_tasks"
    landmarks: list[Landmark] = Field(
        default_factory=list,
        description="33-point landmark list. Empty when source='raw_frame'.",
    )
    raw_frame_b64: str | None = Field(
        None,
        description="Base64-encoded JPEG frame. Required when source='raw_frame'.",
    )


class RepCompleteMessage(AppBaseModel):
    """Client signals that one rep has been completed."""

    type: Literal["REP_COMPLETE"] = "REP_COMPLETE"
    session_id: UUID
    exercise_id: UUID
    rep_number: int = Field(..., ge=1)
    set_number: int = Field(..., ge=1)
    timestamp_ms: int


class PingMessage(AppBaseModel):
    """Client keep-alive ping. Server responds with PongMessage."""

    type: Literal["PING"] = "PING"
    timestamp_ms: int


# Union type for inbound message parsing
InboundMessage = Annotated[
    Union[LandmarkFrame, RepCompleteMessage, PingMessage],
    Field(discriminator="type"),
]


# ═════════════════════════════════════════════════════════════════════════════
# OUTBOUND  (Server → Client)
# ═════════════════════════════════════════════════════════════════════════════

class OverlayPoint(AppResponseModel):
    """
    One landmark to highlight in the frontend's AR overlay.
    The client draws these on top of the live camera feed.
    """

    landmark_id: int = Field(..., ge=0, le=32)
    x: float
    y: float
    highlight: bool = True
    colour: str = Field(
        "#FF4444",
        description="Hex colour for the highlight. Red for errors, amber for warnings.",
    )


class FeedbackMessage(AppResponseModel):
    """
    Real-time form correction message.  Sent when pose_analyzer detects a
    joint angle violation that has persisted for POSE_VIOLATION_FRAME_COUNT
    consecutive frames.
    """

    type: Literal["FEEDBACK"] = "FEEDBACK"
    session_id: UUID
    timestamp_ms: int
    severity: Literal["info", "warning", "error"] = Field(
        ...,
        description=(
            "info: positive cue; "
            "warning: minor deviation, correct soon; "
            "error: significant deviation, correct immediately."
        ),
    )
    message: str = Field(..., description="Patient-facing correction instruction (≤ 20 words).")
    affected_joint: str | None = Field(None, description="MediaPipe joint name that triggered the feedback.")
    error_type: str | None = Field(None, description="Machine-readable violation code from pose_analyzer.")
    actual_angle: float | None = None
    expected_min_angle: float | None = None
    expected_max_angle: float | None = None
    deviation_degrees: float | None = None
    form_score: float | None = Field(None, description="Rolling form quality score at this moment (0–100).")
    overlay_points: list[OverlayPoint] = Field(
        default_factory=list,
        description="Landmarks to highlight in the AR overlay.",
    )
    from_cache: bool = Field(False, description="True if the message text was served from Redis cache.")


class MilestoneMessage(AppResponseModel):
    """Sent when the patient completes a rep or set."""

    type: Literal["MILESTONE"] = "MILESTONE"
    session_id: UUID
    timestamp_ms: int
    milestone_type: Literal["rep", "set"] = "rep"
    rep_number: int | None = None
    set_number: int | None = None
    exercise_id: UUID
    message: str = Field(..., description="Encouraging message, e.g. 'Rep 3 of 10 — great form!'.")
    form_score: float | None = None


class ExerciseDoneMessage(AppResponseModel):
    """Sent when the current exercise is complete. Carries the next exercise details."""

    type: Literal["EXERCISE_DONE"] = "EXERCISE_DONE"
    session_id: UUID
    timestamp_ms: int
    completed_exercise_id: UUID
    completed_exercise_name: str
    next_exercise_id: UUID | None = Field(None, description="None if this was the last exercise.")
    next_exercise_name: str | None = None
    next_exercise_sets: int | None = None
    next_exercise_reps: int | None = None
    rest_seconds: int = Field(30, description="Recommended rest before starting the next exercise.")
    message: str


class SessionSummaryMessage(AppResponseModel):
    """Sent when all exercises in the session are complete."""

    type: Literal["SESSION_SUMMARY"] = "SESSION_SUMMARY"
    session_id: UUID
    timestamp_ms: int
    avg_quality_score: float | None
    completion_pct: float | None
    total_reps: int
    total_sets: int
    duration_seconds: int
    feedback_count: int = Field(0, description="Total correction messages sent during the session.")
    message: str = Field(..., description="AI-generated encouraging summary.")
    plan_adapted: bool = Field(False, description="True if the plan was updated based on this session.")


class RedFlagMessage(AppResponseModel):
    """
    Sent immediately when red_flag_monitor detects a danger condition.
    The client must display this prominently and pause or stop the exercise.
    """

    type: Literal["RED_FLAG"] = "RED_FLAG"
    session_id: UUID
    timestamp_ms: int
    severity: Literal["warn", "stop", "seek_care"] = Field(
        ...,
        description=(
            "warn: continue with caution; "
            "stop: stop this exercise immediately; "
            "seek_care: stop all activity and contact a clinician."
        ),
    )
    message: str = Field(..., description="Patient-facing instruction in plain language.")
    trigger_type: str
    red_flag_id: UUID = Field(..., description="ID of the RedFlagEvent record for tracking.")


class PongMessage(AppResponseModel):
    """Server response to a client PingMessage."""

    type: Literal["PONG"] = "PONG"
    timestamp_ms: int
    server_time_ms: int


class ErrorMessage(AppResponseModel):
    """
    Recoverable protocol error.  The session remains open.
    Fatal errors (auth failure, session not found) close the WebSocket with
    the appropriate close code instead.
    """

    type: Literal["ERROR"] = "ERROR"
    code: str = Field(..., description="Machine-readable error code.")
    message: str
    timestamp_ms: int


# Union type for outbound message serialisation
OutboundMessage = Union[
    FeedbackMessage,
    MilestoneMessage,
    ExerciseDoneMessage,
    SessionSummaryMessage,
    RedFlagMessage,
    PongMessage,
    ErrorMessage,
]


# ── WebSocket close codes (RFC 6455 custom range 4000–4999) ───────────────────

class WSCloseCode:
    AUTH_FAILED = 4001
    SESSION_NOT_FOUND = 4004
    SESSION_ALREADY_ACTIVE = 4009
    RATE_LIMITED = 4029
    SERVER_ERROR = 4500