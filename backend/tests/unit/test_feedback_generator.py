"""
Unit tests for app/services/feedback_generator.py.

Tests Redis cache hit/miss behaviour, Claude fallback on API failure,
and static fallback on total failure.  No real Redis or Claude calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.feedback_generator import FeedbackGeneratorService, _FALLBACKS, _DEFAULT_FALLBACK
from app.services.pose_analyzer import JointViolation
from app.core.exceptions import FeedbackGenerationError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _violation(
    error_type: str = "ankle_insufficient_range",
    joint: str = "left_ankle",
    severity: str = "warning",
    deviation: float = -8.0,
) -> JointViolation:
    return JointViolation(
        joint=joint,
        actual_angle=7.0,
        min_angle=10.0,
        max_angle=35.0,
        deviation_degrees=deviation,
        deviation_direction="flexed",
        error_type=error_type,
        severity=severity,
        overlay_landmark_ids=[25, 27, 31],
    )


def _make_svc(redis=None, claude=None) -> FeedbackGeneratorService:
    redis  = redis  or AsyncMock()
    claude = claude or AsyncMock()
    return FeedbackGeneratorService(claude, redis)


# ── Cache hit ─────────────────────────────────────────────────────────────────

class TestCacheHit:

    @pytest.mark.asyncio
    async def test_returns_cached_message_without_claude(self):
        cached_msg = "Push your ankle further for a better range."
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=cached_msg)

        claude = AsyncMock()
        svc    = _make_svc(redis=redis, claude=claude)
        db     = AsyncMock()

        msg, from_cache = await svc.get_feedback(
            violation=_violation(),
            exercise_slug="seated-ankle-circles",
            exercise_name="Seated Ankle Circles",
            difficulty="beginner",
            db=db,
        )

        assert msg == cached_msg
        assert from_cache is True
        claude.generate_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_key_includes_slug_error_type_difficulty(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value="cached")
        svc = _make_svc(redis=redis)

        await svc.get_feedback(
            violation=_violation(error_type="knee_valgus"),
            exercise_slug="calf-raises",
            exercise_name="Calf Raises",
            difficulty="intermediate",
            db=AsyncMock(),
        )

        call_args = redis.get.call_args[0][0]
        assert "calf-raises" in call_args
        assert "knee_valgus" in call_args
        assert "intermediate" in call_args


# ── Cache miss → Claude ───────────────────────────────────────────────────────

class TestCacheMiss:

    @pytest.mark.asyncio
    async def test_calls_claude_on_cache_miss(self):
        redis  = AsyncMock()
        redis.get    = AsyncMock(return_value=None)
        redis.setex  = AsyncMock()

        claude = AsyncMock()
        claude.generate_feedback = AsyncMock(
            return_value="Push your ankle a little further."
        )

        svc = _make_svc(redis=redis, claude=claude)
        db  = AsyncMock()

        msg, from_cache = await svc.get_feedback(
            violation=_violation(),
            exercise_slug="seated-ankle-circles",
            exercise_name="Seated Ankle Circles",
            difficulty="beginner",
            db=db,
        )

        assert msg == "Push your ankle a little further."
        assert from_cache is False
        claude.generate_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_caches_claude_response(self):
        redis = AsyncMock()
        redis.get   = AsyncMock(return_value=None)
        redis.setex = AsyncMock()

        claude = AsyncMock()
        claude.generate_feedback = AsyncMock(return_value="Move further.")

        svc = _make_svc(redis=redis, claude=claude)
        await svc.get_feedback(
            violation=_violation(),
            exercise_slug="ex-slug",
            exercise_name="Exercise",
            difficulty="beginner",
            db=AsyncMock(),
        )

        assert redis.setex.called
        call_args = redis.setex.call_args
        assert "Move further." in call_args[0] or "Move further." in str(call_args)


# ── Fallbacks ─────────────────────────────────────────────────────────────────

class TestFallbacks:

    @pytest.mark.asyncio
    async def test_static_fallback_on_claude_failure(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)

        claude = AsyncMock()
        claude.generate_feedback = AsyncMock(
            side_effect=FeedbackGenerationError("API down")
        )

        svc = _make_svc(redis=redis, claude=claude)
        msg, from_cache = await svc.get_feedback(
            violation=_violation(error_type="ankle_insufficient_range"),
            exercise_slug="ex",
            exercise_name="Exercise",
            difficulty="beginner",
            db=AsyncMock(),
        )

        # Should return the known static fallback, not raise
        assert msg == _FALLBACKS["ankle_insufficient_range"]
        assert from_cache is False

    @pytest.mark.asyncio
    async def test_default_fallback_for_unknown_error_type(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)

        claude = AsyncMock()
        claude.generate_feedback = AsyncMock(
            side_effect=FeedbackGenerationError("API down")
        )

        svc = _make_svc(redis=redis, claude=claude)
        msg, _ = await svc.get_feedback(
            violation=_violation(error_type="completely_unknown_error_xyz"),
            exercise_slug="ex",
            exercise_name="Exercise",
            difficulty="beginner",
            db=AsyncMock(),
        )

        assert msg == _DEFAULT_FALLBACK

    @pytest.mark.asyncio
    async def test_all_known_error_types_have_fallbacks(self):
        """Every error_type in _FALLBACKS must be a non-empty string."""
        for error_type, message in _FALLBACKS.items():
            assert isinstance(message, str)
            assert len(message.strip()) > 0, f"Empty fallback for {error_type}"


# ── Warm cache ────────────────────────────────────────────────────────────────

class TestWarmCache:

    @pytest.mark.asyncio
    async def test_warm_cache_skips_existing_keys(self):
        redis = AsyncMock()
        redis.exists = AsyncMock(return_value=True)   # already cached

        claude = AsyncMock()
        svc    = _make_svc(redis=redis, claude=claude)

        exercises = [
            {"slug": "ankle-circles", "name": "Ankle Circles",
             "difficulty": "beginner", "landmark_rules": {}},
        ]

        await svc.warm_cache(exercises=exercises, db=AsyncMock())

        # Claude should NOT be called since all keys already exist
        claude.generate_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_cache_populates_missing_keys(self):
        redis = AsyncMock()
        redis.exists = AsyncMock(return_value=False)   # nothing cached
        redis.setex  = AsyncMock()

        claude = AsyncMock()
        claude.generate_feedback = AsyncMock(return_value="Good correction.")

        svc       = _make_svc(redis=redis, claude=claude)
        exercises = [
            {"slug": "ankle-circles", "name": "Ankle Circles",
             "difficulty": "beginner", "landmark_rules": {}},
        ]

        await svc.warm_cache(exercises=exercises, db=AsyncMock())

        # Claude should be called for the common error types
        assert claude.generate_feedback.call_count > 0