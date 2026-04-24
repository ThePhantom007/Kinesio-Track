"""
Custom exception hierarchy for Kinesio-Track.

Design rules:
  - Every domain error is a subclass of KinesioBaseError.
  - Each exception carries an ``error_code`` string (snake_case) so the
    frontend can branch on it without parsing message text.
  - HTTP status mapping lives here; FastAPI exception handlers in main.py
    read ``http_status`` to produce consistent error responses.
  - Never import FastAPI here — this module must be importable by Celery
    workers that don't load the HTTP layer.
"""

from __future__ import annotations

from typing import Any


class KinesioBaseError(Exception):
    """Root for all application errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str = "", detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail  # optional structured context sent to the client

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.error_code!r}, msg={self.message!r})"


# ── Authentication & Authorisation ────────────────────────────────────────────

class AuthenticationError(KinesioBaseError):
    """Bad or missing credentials."""
    http_status = 401
    error_code = "authentication_error"


class TokenExpiredError(AuthenticationError):
    """JWT has passed its expiry time."""
    error_code = "token_expired"


class PermissionDeniedError(KinesioBaseError):
    """Authenticated user lacks the required role/ownership."""
    http_status = 403
    error_code = "permission_denied"


# ── Resource errors ───────────────────────────────────────────────────────────

class NotFoundError(KinesioBaseError):
    """Requested resource does not exist."""
    http_status = 404
    error_code = "not_found"


class ConflictError(KinesioBaseError):
    """Resource already exists or state conflict."""
    http_status = 409
    error_code = "conflict"


class ValidationError(KinesioBaseError):
    """Request payload failed Pydantic validation (unhandled by FastAPI)."""
    http_status = 422
    error_code = "validation_error"


# ── Plan & AI ─────────────────────────────────────────────────────────────────

class PlanGenerationError(KinesioBaseError):
    """Claude failed to generate an exercise plan after max retries."""
    http_status = 502
    error_code = "plan_generation_error"


class PlanValidationError(KinesioBaseError):
    """
    Claude returned a plan that doesn't match the required JSON schema.
    ``detail`` contains a diff of the missing/invalid fields so the
    retry prompt can be constructed with targeted corrections.
    """
    http_status = 502
    error_code = "plan_validation_error"


class PlanAdaptationError(KinesioBaseError):
    """Claude failed to produce a valid JSON Patch for plan adaptation."""
    http_status = 502
    error_code = "plan_adaptation_error"


class FeedbackGenerationError(KinesioBaseError):
    """Claude failed to generate a correction message."""
    http_status = 502
    error_code = "feedback_generation_error"


# ── Pose Analysis ─────────────────────────────────────────────────────────────

class PoseAnalysisError(KinesioBaseError):
    """
    Landmark rules engine encountered an unexpected error.
    Distinct from a normal form violation — this means the *analyzer itself*
    crashed, e.g. missing landmark index, corrupt rules' payload.
    """
    http_status = 500
    error_code = "pose_analysis_error"


class InsufficientLandmarksError(PoseAnalysisError):
    """Too many landmarks have visibility below the configured threshold."""
    error_code = "insufficient_landmarks"


# ── Red Flag ──────────────────────────────────────────────────────────────────

class RedFlagError(KinesioBaseError):
    """
    Raised when the red-flag escalation pipeline itself fails
    (not the same as a *successful* red-flag detection).
    """
    http_status = 500
    error_code = "red_flag_error"


# ── Video Processing ──────────────────────────────────────────────────────────

class VideoProcessingError(KinesioBaseError):
    """MediaPipe or OpenCV error during video analysis."""
    http_status = 500
    error_code = "video_processing_error"


class VideoDownloadError(VideoProcessingError):
    """Failed to retrieve the video from S3."""
    error_code = "video_download_error"


# ── Session ───────────────────────────────────────────────────────────────────

class SessionNotFoundError(NotFoundError):
    """No active session for the given session_id."""
    error_code = "session_not_found"


class SessionAlreadyActiveError(ConflictError):
    """Patient already has an in-progress session."""
    error_code = "session_already_active"


class SessionNotActiveError(KinesioBaseError):
    """Attempted to perform an operation on a session that is not in-progress."""
    http_status = 400
    error_code = "session_not_active"


# ── Analytics ─────────────────────────────────────────────────────────────────

class InsufficientDataError(KinesioBaseError):
    """
    Not enough session history to compute a meaningful metric
    (e.g. recovery ETA requires at least N sessions).
    """
    http_status = 422
    error_code = "insufficient_data"


# ── External services ─────────────────────────────────────────────────────────

class ExternalServiceError(KinesioBaseError):
    """A third-party service (S3, FCM, SMTP) returned an unexpected error."""
    http_status = 502
    error_code = "external_service_error"


class RateLimitExceededError(KinesioBaseError):
    """Client has exceeded the configured rate limit for this endpoint."""
    http_status = 429
    error_code = "rate_limit_exceeded"

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60) -> None:
        super().__init__(message, detail={"retry_after_seconds": retry_after})
        self.retry_after = retry_after