"""
Google Gemini SDK wrapper — drop-in replacement for the Anthropic ClaudeClient.
Public interface is identical so all services work without changes.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from google import genai
from google.genai import types as genai_types
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
)
from app.core.logging import get_logger
from app.models.token_usage import AICallType
from app.schemas.plan import ExercisePlanAIOutput

log = get_logger(__name__)

_RETRY_DELAYS = [1.0, 3.0, 7.0]


class ClaudeClient:
    """
    Gemini wrapper with identical public interface to the original ClaudeClient.
    Named ClaudeClient to avoid changing all imports across the codebase.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._cost_tracker = CostTracker()
        log.info("gemini_client_initialised", model=settings.GEMINI_MODEL)

    async def generate_initial_plan(
        self,
        ctx: IntakeContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> ExercisePlanAIOutput:
        user_prompt = build_initial_plan_prompt(ctx)
        last_error: Exception | None = None

        for attempt in range(settings.GEMINI_MAX_RETRIES):
            try:
                with timer() as t:
                    raw, usage = await self._call(
                        system=INITIAL_PLAN_SYSTEM_PROMPT,
                        user=user_prompt,
                        max_tokens=settings.GEMINI_PLAN_MAX_TOKENS,
                    )
                plan = validate_initial_plan(raw)
                await self._cost_tracker.record(
                    db,
                    call_type=AICallType.INITIAL_PLAN,
                    model=settings.GEMINI_MODEL,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    patient_id=patient_id,
                    retry_count=attempt,
                    latency_ms=t.elapsed_ms,
                )
                return plan
            except PlanValidationError as exc:
                last_error = exc
                log.warning("initial_plan_validation_failed", attempt=attempt + 1, error=str(exc))
                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    user_prompt = build_correction_prompt(user_prompt, exc)
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            except Exception as exc:
                last_error = exc
                log.error("gemini_api_error", attempt=attempt + 1, error=str(exc))
                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        raise PlanGenerationError(
            f"Failed to generate a valid exercise plan after {settings.GEMINI_MAX_RETRIES} attempts.",
            detail={"last_error": str(last_error)},
        )

    async def adapt_plan(
        self,
        ctx: AdaptationContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        user_prompt = build_adapt_prompt(ctx)
        last_error: Exception | None = None

        for attempt in range(settings.GEMINI_MAX_RETRIES):
            try:
                with timer() as t:
                    raw, usage = await self._call(
                        system=ADAPT_PLAN_SYSTEM_PROMPT,
                        user=user_prompt,
                        max_tokens=20480,
                    )
                patch = validate_plan_patch(raw)
                await self._cost_tracker.record(
                    db,
                    call_type=AICallType.ADAPT_PLAN,
                    model=settings.GEMINI_MODEL,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    patient_id=patient_id,
                    retry_count=attempt,
                    latency_ms=t.elapsed_ms,
                )
                return patch
            except PlanValidationError as exc:
                last_error = exc
                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    user_prompt = build_correction_prompt(user_prompt, exc)
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            except Exception as exc:
                last_error = exc
                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        raise PlanAdaptationError(
            f"Failed to produce a valid plan patch after {settings.GEMINI_MAX_RETRIES} attempts.",
            detail={"last_error": str(last_error)},
        )

    async def escalate_red_flag(
        self,
        ctx: RedFlagContext,
        db: AsyncSession,
        patient_id: UUID | None = None,
    ) -> dict[str, Any]:
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
                model=settings.GEMINI_MODEL,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                patient_id=patient_id,
                latency_ms=t.elapsed_ms,
            )
            return response
        except Exception as exc:
            log.error("red_flag_escalation_failed", error=str(exc))
            return {
                "severity": "stop",
                "immediate_action": (
                    "Please stop the exercise and rest. "
                    "Your clinician has been notified."
                ),
                "clinician_note": (
                    f"Automated red-flag escalation failed ({exc}). "
                    "Manual review required."
                ),
                "session_recommendation": "stop_session",
            }

    async def generate_feedback(
        self,
        ctx: FeedbackContext,
        db: AsyncSession,
        session_id: UUID | None = None,
    ) -> str:
        user_prompt = build_feedback_prompt(ctx)
        try:
            with timer() as t:
                raw, usage = await self._call(
                    system=FEEDBACK_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=settings.GEMINI_FEEDBACK_MAX_TOKENS,
                )
            message = validate_feedback_message(raw)
            await self._cost_tracker.record(
                db,
                call_type=AICallType.FEEDBACK,
                model=settings.GEMINI_MODEL,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                session_id=session_id,
                latency_ms=t.elapsed_ms,
            )
            return message
        except Exception as exc:
            log.error("feedback_generation_failed", error=str(exc))
            raise FeedbackGenerationError(
                f"Gemini API error during feedback generation: {exc}",
            ) from exc

    async def _call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
    ) -> tuple[str, dict]:
        """
        Execute one Gemini generate_content call asynchronously.
        Returns (response_text, usage_dict).
        """
        loop = asyncio.get_running_loop()

        def _sync_call():
            return self._client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_budget=0,  # disable thinking for faster responses
                    ),
                ),
            )

        response = await loop.run_in_executor(None, _sync_call)

        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            log.info("gemini_finish_reason", reason=str(finish_reason))

        log.info(
            "gemini_usage",
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
            output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
            max_tokens=max_tokens,
        )

        text = response.text or ""
        usage = {
            "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
            "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        }
        return text, usage

    async def close(self) -> None:
        """No-op — Gemini client has no persistent connection to close."""
        log.info("gemini_client_closed")