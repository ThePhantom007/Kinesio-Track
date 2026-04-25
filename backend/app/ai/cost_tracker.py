"""
Records Claude API token usage and computes estimated cost per call.

Called by claude_client after every API response (success or validated retry).
Writes a TokenUsage row asynchronously so it never blocks the caller.

Pricing table
-------------
Prices are approximate and should be updated when Anthropic revises them.
The tracker uses the model string returned in the API response to look up
the correct rate.  Unknown models fall back to a conservative estimate.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.token_usage import AICallType, TokenUsage

log = get_logger(__name__)

# ── Pricing table (USD per 1M tokens) ────────────────────────────────────────
# Update these when Anthropic revises pricing.

_INPUT_PRICE_PER_M: dict[str, float] = {
    "claude-opus-4-20250514":   15.00,
    "claude-sonnet-4-20250514":  3.00,
    "claude-haiku-4-5-20251001": 0.80,
}
_OUTPUT_PRICE_PER_M: dict[str, float] = {
    "claude-opus-4-20250514":   75.00,
    "claude-sonnet-4-20250514": 15.00,
    "claude-haiku-4-5-20251001": 4.00,
}
_FALLBACK_INPUT_PRICE  = 3.00
_FALLBACK_OUTPUT_PRICE = 15.00


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD for a single API call."""
    in_rate  = _INPUT_PRICE_PER_M.get(model, _FALLBACK_INPUT_PRICE)
    out_rate = _OUTPUT_PRICE_PER_M.get(model, _FALLBACK_OUTPUT_PRICE)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ── Main tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Async cost tracker.  One instance is shared across the application
    (created in claude_client, injected via dependency or passed explicitly).

    Usage::

        tracker = CostTracker()
        await tracker.record(
            db=db,
            call_type=AICallType.INITIAL_PLAN,
            model="claude-sonnet-4-20250514",
            input_tokens=1200,
            output_tokens=850,
            patient_id=patient.id,
            latency_ms=4200
        )
    """

    async def record(
        self,
        db: AsyncSession,
        *,
        call_type: AICallType,
        model: str,
        input_tokens: int,
        output_tokens: int,
        patient_id: UUID | None = None,
        session_id: UUID | None = None,
        was_cached: bool = False,
        retry_count: int = 0,
        validation_passed: bool = True,
        latency_ms: int | None = None,
    ) -> None:
        """
        Write a TokenUsage row.  Failures are logged but never re-raised —
        cost tracking must not break the primary request flow.
        """
        cost = _estimate_cost(model, input_tokens, output_tokens)

        usage = TokenUsage(
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            patient_id=patient_id,
            session_id=session_id,
            was_cached=was_cached,
            retry_count=retry_count,
            validation_passed=validation_passed,
            latency_ms=latency_ms,
        )

        try:
            db.add(usage)
            await db.flush()   # flush without committing — caller owns the transaction
            log.info(
                "token_usage_recorded",
                call_type=call_type.value,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost, 6),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            log.error("token_usage_record_failed", error=str(exc), call_type=call_type.value)

    async def check_budget(self, db: AsyncSession) -> dict[str, Any]:
        """
        Query the monthly token spend and return a summary dict.
        Used by the /admin/ai-costs endpoint and the daily budget-check task.

        Returns dict with keys: total_cost_usd, budget_usd, remaining_usd,
        percent_used, by_call_type.
        """
        from sqlalchemy import extract, func, select
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        result = await db.execute(
            select(
                TokenUsage.call_type,
                func.sum(TokenUsage.input_tokens).label("input_tokens"),
                func.sum(TokenUsage.output_tokens).label("output_tokens"),
                func.sum(TokenUsage.cost_usd).label("cost_usd"),
                func.count().label("call_count"),
            )
            .where(
                extract("year",  TokenUsage.called_at) == now.year,
                extract("month", TokenUsage.called_at) == now.month,
            )
            .group_by(TokenUsage.call_type)
        )
        rows = result.all()

        by_call_type: dict[str, dict] = {}
        total_cost = 0.0
        for row in rows:
            cost = float(row.cost_usd or 0)
            total_cost += cost
            by_call_type[row.call_type] = {
                "input_tokens":  int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "cost_usd":      round(cost, 4),
                "call_count":    int(row.call_count),
            }

        budget = settings.MONTHLY_TOKEN_BUDGET_USD
        return {
            "total_cost_usd":  round(total_cost, 4),
            "budget_usd":      budget,
            "remaining_usd":   round(budget - total_cost, 4),
            "percent_used":    round(total_cost / budget * 100, 1) if budget > 0 else 0.0,
            "by_call_type":    by_call_type,
            "month":           now.strftime("%Y-%m"),
        }


# ── Timing context manager ────────────────────────────────────────────────────

class _Timer:
    """Simple context manager for measuring wall-clock latency in ms."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)


def timer() -> _Timer:
    """Return a new _Timer context manager."""
    return _Timer()