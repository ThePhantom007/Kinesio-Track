"""
Converts a pose_analyzer JointViolation into a patient-facing correction
message string, using Redis as a cache to avoid Claude API calls for
repeated (exercise, error_type) combinations.

Cache strategy
--------------
Key:   feedback_cache:{exercise_slug}:{error_type}:{difficulty}
TTL:   REDIS_FEEDBACK_CACHE_TTL (default 24 h)
Hit:   return cached string, record was_cached=True in token_usage
Miss:  call claude_client.generate_feedback(), write to cache, return

The feedback string is a single sentence ≤ 20 words (enforced by
response_parser.validate_feedback_message).

Fallback hierarchy
------------------
  1. Redis cache hit                → return immediately
  2. Claude API call               → cache + return
  3. Claude fails                  → generic fallback from _FALLBACKS dict
  4. No matching fallback          → hardcoded safe default sentence

The WebSocket handler must never block waiting for a slow Claude call.
This service is awaited inline but the fallback ensures forward progress
even under API degradation.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.claude_client import ClaudeClient
from app.ai.prompt_templates.feedback import FeedbackContext, feedback_cache_key
from app.core.config import settings
from app.core.exceptions import FeedbackGenerationError
from app.core.logging import get_logger
from app.services.pose_analyzer import JointViolation

log = get_logger(__name__)

# ── Generic fallbacks keyed by error_type ────────────────────────────────────
# Used when Claude is unavailable. Written by a physiotherapist.

_FALLBACKS: dict[str, str] = {
    "knee_valgus":               "Push your knee outward to align with your toes.",
    "knee_insufficient_flexion": "Bend your knee a little more to reach the full range.",
    "knee_hyperflexion":         "Ease off slightly — bring your knee back to a comfortable bend.",
    "lumbar_hyperextension":     "Gently flatten your lower back and engage your core.",
    "lumbar_flexion":            "Lift your chest and keep your spine long and tall.",
    "shoulder_elevation":        "Relax your shoulders down away from your ears.",
    "shoulder_hyperflexion":     "Lower your arm slightly to stay within the safe range.",
    "ankle_insufficient_range":  "Push your ankle a little further to reach full range.",
    "ankle_hyperflexion":        "Ease off your ankle — hold at the comfortable end range.",
    "bilateral_asymmetry":       "Distribute your weight evenly across both sides.",
    "neck_hyperflexion":         "Gently bring your chin back to a neutral position.",
    "neck_hyperextension":       "Lower your chin slightly and lengthen the back of your neck.",
    "hip_insufficient_flexion":  "Hinge a little deeper at your hip to reach the target range.",
    "elbow_hyperextension":      "Keep a soft bend in your elbow at the top of the movement.",
    "insufficient_range":        "Move a little further to reach the full range of motion.",
    "excessive_range":           "Ease back slightly — you are moving beyond the safe range.",
}

_DEFAULT_FALLBACK = "Move slowly and carefully, staying within a comfortable range."


class FeedbackGeneratorService:

    def __init__(self, claude_client: ClaudeClient, redis) -> None:
        self._claude = claude_client
        self._redis  = redis

    async def get_feedback(
        self,
        *,
        violation: JointViolation,
        exercise_slug: str,
        exercise_name: str,
        difficulty: str,
        db: AsyncSession,
        session_id: Any = None,
        patient_age: int | None = None,
    ) -> tuple[str, bool]:
        """
        Return a correction message for the given violation.

        Args:
            violation:     JointViolation from pose_analyzer.
            exercise_slug: Stable exercise identifier for the cache key.
            exercise_name: Human-readable name passed to the Claude prompt.
            difficulty:    "beginner" | "intermediate" | "advanced"
            db:            Async DB session for cost_tracker writes.
            session_id:    Optional session UUID for token_usage attribution.
            patient_age:   Optional age for prompt calibration.

        Returns:
            (message_string, from_cache) tuple.
        """
        cache_key = feedback_cache_key(exercise_slug, violation.error_type, difficulty)

        # 1. Cache hit
        cached = await self._redis.get(cache_key)
        if cached:
            log.debug(
                "feedback_cache_hit",
                exercise=exercise_slug,
                error_type=violation.error_type,
            )
            return cached, True

        # 2. Claude call
        ctx = FeedbackContext(
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            error_type=violation.error_type,
            affected_joint=violation.joint,
            deviation_degrees=abs(violation.deviation_degrees),
            deviation_direction=violation.deviation_direction,
            difficulty=difficulty,
            patient_age=patient_age,
        )

        try:
            message = await self._claude.generate_feedback(ctx, db, session_id=session_id)
            await self._redis.setex(
                cache_key,
                settings.REDIS_FEEDBACK_CACHE_TTL,
                message,
            )
            log.debug(
                "feedback_generated_and_cached",
                exercise=exercise_slug,
                error_type=violation.error_type,
            )
            return message, False

        except FeedbackGenerationError as exc:
            log.warning(
                "feedback_claude_failed_using_fallback",
                error=str(exc),
                error_type=violation.error_type,
            )

        # 3. Static fallback
        fallback = _FALLBACKS.get(violation.error_type, _DEFAULT_FALLBACK)
        return fallback, False

    async def warm_cache(
        self,
        *,
        exercises: list[dict],
        db: AsyncSession,
    ) -> None:
        """
        Pre-warm the feedback cache for all exercises in a session at startup.
        Called by session_manager when a session begins so the first violation
        in a session always hits the cache.

        Only warms entries that are not already cached.

        Args:
            exercises: List of exercise dicts with keys:
                       slug, name, difficulty, landmark_rules.
            db:        DB session for any cost_tracker writes.
        """
        common_errors = [
            ("knee_valgus",           "left_knee",  "flexed"),
            ("lumbar_hyperextension", "lumbar_spine", "extended"),
            ("shoulder_elevation",    "left_shoulder", "extended"),
            ("bilateral_asymmetry",   "knee",        "asymmetric"),
            ("insufficient_range",    "left_ankle",  "flexed"),
        ]

        for ex in exercises:
            slug       = ex.get("slug", "")
            name       = ex.get("name", "")
            difficulty = ex.get("difficulty", "beginner")

            for error_type, joint, direction in common_errors:
                key = feedback_cache_key(slug, error_type, difficulty)
                if await self._redis.exists(key):
                    continue

                ctx = FeedbackContext(
                    exercise_name=name,
                    exercise_slug=slug,
                    error_type=error_type,
                    affected_joint=joint,
                    deviation_degrees=15.0,
                    deviation_direction=direction,
                    difficulty=difficulty,
                )
                try:
                    message = await self._claude.generate_feedback(ctx, db)
                    await self._redis.setex(key, settings.REDIS_FEEDBACK_CACHE_TTL, message)
                except FeedbackGenerationError:
                    pass   # warm-up failures are non-fatal