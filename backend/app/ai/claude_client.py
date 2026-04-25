"""
Async Anthropic SDK wrapper.  Single entry point for all Claude API calls.

Public interface
----------------
  generate_initial_plan(ctx, db, patient_id)   → ExercisePlanAIOutput
  adapt_plan(ctx, db, patient_id)              → list[dict]  (JSON Patch)
  escalate_red_flag(ctx, db, patient_id)       → dict
  generate_feedback(ctx, db, session_id)       → str

Design decisions
----------------
- Each method has its own token budget, system prompt, and retry behaviour.
- Retries are validation-aware: on PlanValidationError the corrective prompt
  (from response_parser.build_correction_prompt) replaces the user turn so
  Claude sees exactly which fields are wrong.
- The cost_tracker records usage after every successful final response
  (not per retry attempt).
- All methods are fully async — never block the event loop.
- The client is instantiated once at app startup and shared via the FastAPI
  app state (app.state.claude_client).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.cost_tracker import CostTracker, timer
from app.ai.prompt_templates.adapt_plan import (
    ADAPT_PLAN_SYSTEM_PROMPT,
    AdaptationContext,
    build_adapt_prompt,
)
from app.ai.prompt_templates.feedback import (
    FEEDBACK_SYSTEM_PROMPT,
    FeedbackContext,
    build_feedback_prompt,
)
from app.ai.prompt_templates.initial_plan import (
    INITIAL_PLAN_SYSTEM_PROMPT,
    IntakeContext,
    build_initial_plan_prompt,
)
from app.ai.prompt_templates.red_flag import (
    RED_FLAG_SYSTEM_PROMPT,
    RedFlagContext,
    build_red_flag_prompt,
)
from app.ai.response_parser import (
    build_correction_prompt,
    validate_feedback_message,
    validate_initial_plan,
    validate_plan_patch,
    validate_red_flag_response,
)
from app.core.config import settings
from app.core.exceptions import (
    FeedbackGenerationError,
    PlanAdaptationError,
    PlanGenerationError,
    PlanValidationError,
    RedFlagError,
)
from app.core.logging import get_logger
from app.models.token_usage import AICallType
from app.schemas.plan import ExercisePlanAIOutput

log = get_logger(__name__)

# ── Retry config per call type ────────────────────────────────────────────────

_RETRY_DELAYS = [1.0, 3.0, 7.0]   # seconds between attempts (exponential-ish)


class ClaudeClient:
    """
    Async wrapper around the Anthropic SDK.

    Instantiate once and share via app.state::

        app.state.claude_client = ClaudeClient()

    Each public method acquires its own HTTP connection from the SDK's
    internal connection pool.
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.ANTHROPIC_TIMEOUT_SECONDS,
            max_retries=0,   # We manage retries ourselves for validation awareness
        )
        self._cost_tracker = CostTracker()
        log.info("claude_client_initialised", model=settings.ANTHROPIC_MODEL)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Initial plan generation
    # ═════════════════════════════════════════════════════════════════════════

    async def generate_initial_plan(
        self,
        ctx: IntakeContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> ExercisePlanAIOutput:
        """
        Generate a full physiotherapy exercise plan from an IntakeContext.

        Retries up to ANTHROPIC_MAX_RETRIES times on PlanValidationError,
        passing the field-level diff back to Claude each time.

        Returns:
            Validated ExercisePlanAIOutput.

        Raises:
            PlanGenerationError: All retry attempts exhausted.
        """
        user_prompt = build_initial_plan_prompt(ctx)
        last_error: Exception | None = None

        for attempt in range(settings.ANTHROPIC_MAX_RETRIES):
            try:
                with timer() as t:
                    raw, usage = await self._call(
                        system=INITIAL_PLAN_SYSTEM_PROMPT,
                        user=user_prompt,
                        max_tokens=settings.ANTHROPIC_PLAN_MAX_TOKENS,
                    )

                plan = validate_initial_plan(raw)

                await self._cost_tracker.record(
                    db,
                    call_type=AICallType.INITIAL_PLAN,
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    patient_id=patient_id,
                    retry_count=attempt,
                    latency_ms=t.elapsed_ms,
                )
                return plan

            except PlanValidationError as exc:
                last_error = exc
                log.warning(
                    "initial_plan_validation_failed",
                    attempt=attempt + 1,
                    max_attempts=settings.ANTHROPIC_MAX_RETRIES,
                    error=str(exc),
                )
                if attempt < settings.ANTHROPIC_MAX_RETRIES - 1:
                    user_prompt = build_correction_prompt(user_prompt, exc)
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

            except anthropic.APIError as exc:
                last_error = exc
                log.error("anthropic_api_error", attempt=attempt + 1, error=str(exc))
                if attempt < settings.ANTHROPIC_MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        raise PlanGenerationError(
            f"Failed to generate a valid exercise plan after {settings.ANTHROPIC_MAX_RETRIES} attempts.",
            detail={"last_error": str(last_error)},
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Plan adaptation
    # ═════════════════════════════════════════════════════════════════════════

    async def adapt_plan(
        self,
        ctx: AdaptationContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Produce an RFC 6902 JSON Patch array for post-session plan adaptation.

        Returns:
            List of patch operations (may be empty if no adaptation needed).

        Raises:
            PlanAdaptationError: All retry attempts exhausted.
        """
        user_prompt = build_adapt_prompt(ctx)
        last_error: Exception | None = None

        for attempt in range(settings.ANTHROPIC_MAX_RETRIES):
            try:
                with timer() as t:
                    raw, usage = await self._call(
                        system=ADAPT_PLAN_SYSTEM_PROMPT,
                        user=user_prompt,
                        max_tokens=512,
                    )

                patch = validate_plan_patch(raw)

                await self._cost_tracker.record(
                    db,
                    call_type=AICallType.ADAPT_PLAN,
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    patient_id=patient_id,
                    retry_count=attempt,
                    latency_ms=t.elapsed_ms,
                )
                log.info("plan_adapted", op_count=len(patch), attempt=attempt + 1)
                return patch

            except PlanValidationError as exc:
                last_error = exc
                log.warning("adapt_plan_validation_failed", attempt=attempt + 1, error=str(exc))
                if attempt < settings.ANTHROPIC_MAX_RETRIES - 1:
                    user_prompt = build_correction_prompt(user_prompt, exc)
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

            except anthropic.APIError as exc:
                last_error = exc
                log.error("anthropic_api_error", call="adapt_plan", attempt=attempt + 1, error=str(exc))
                if attempt < settings.ANTHROPIC_MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        raise PlanAdaptationError(
            f"Failed to produce a valid plan patch after {settings.ANTHROPIC_MAX_RETRIES} attempts.",
            detail={"last_error": str(last_error)},
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Red-flag escalation
    # ═════════════════════════════════════════════════════════════════════════

    async def escalate_red_flag(
        self,
        ctx: RedFlagContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> dict[str, Any]:
        """
        Generate an escalation response for a detected red-flag condition.

        This call is time-sensitive (patient is mid-session); it does NOT
        retry on validation failure — instead it falls back to a safe default
        response to avoid delaying the patient-facing message.

        Returns:
            Dict with keys: severity, immediate_action, clinician_note,
            session_recommendation.
        """
        user_prompt = build_red_flag_prompt(ctx)

        try:
            with timer() as t:
                raw, usage = await self._call(
                    system=RED_FLAG_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=300,
                )
            response = validate_red_flag_response(raw)

            await self._cost_tracker.record(
                db,
                call_type=AICallType.RED_FLAG,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                patient_id=patient_id,
                latency_ms=t.elapsed_ms,
            )
            return response

        except (PlanValidationError, anthropic.APIError) as exc:
            log.error("red_flag_escalation_failed", error=str(exc), trigger=ctx.trigger_type)
            # Safe fallback — always tell the patient to stop and rest.
            return {
                "severity": "stop",
                "immediate_action": (
                    "Please stop the exercise and rest. "
                    "Your clinician has been notified."
                ),
                "clinician_note": (
                    f"Automated red-flag escalation failed ({exc}). "
                    f"Manual review required. Trigger: {ctx.trigger_type}."
                ),
                "session_recommendation": "stop_session",
            }

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Real-time feedback
    # ═════════════════════════════════════════════════════════════════════════

    async def generate_feedback(
        self,
        ctx: FeedbackContext,
        db: AsyncSession,
        session_id: UUID | None = None,
    ) -> str:
        """
        Generate a single correction message for a real-time form violation.

        Optimised for minimum latency (max_tokens=80).  Does NOT retry —
        the feedback_generator service has already checked the Redis cache;
        if Claude fails here, the caller falls back to a generic message.

        Returns:
            Validated correction sentence (≤ 20 words).

        Raises:
            FeedbackGenerationError: API error (not validation — validation
            failures fall back to a generic message via validate_feedback_message).
        """
        user_prompt = build_feedback_prompt(ctx)

        try:
            with timer() as t:
                raw, usage = await self._call(
                    system=FEEDBACK_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=settings.ANTHROPIC_FEEDBACK_MAX_TOKENS,
                )

            message = validate_feedback_message(raw)

            await self._cost_tracker.record(
                db,
                call_type=AICallType.FEEDBACK,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                session_id=session_id,
                latency_ms=t.elapsed_ms,
            )
            return message

        except anthropic.APIError as exc:
            log.error("feedback_generation_failed", error=str(exc), exercise=ctx.exercise_slug)
            raise FeedbackGenerationError(
                f"Claude API error during feedback generation: {exc}",
            ) from exc

    # ═════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═════════════════════════════════════════════════════════════════════════

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
    ) -> tuple[str, anthropic.types.Usage]:
        """
        Execute one Anthropic messages.create call.

        Returns:
            (response_text, usage_object) tuple.

        Raises:
            anthropic.APIError on any HTTP or SDK-level failure.
        """
        response = await self._client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return text, response.usage

    async def close(self) -> None:
        """Close the underlying HTTP client.  Call during app shutdown."""
        await self._client.close()
        log.info("claude_client_closed")