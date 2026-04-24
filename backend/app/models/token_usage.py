"""
Records every Claude API call for cost monitoring and budget enforcement.

Rows are written by the cost_tracker service immediately after each Claude
response.  The /admin/ai-costs endpoint aggregates this table.

The Celery analytics_tasks.refresh_recovery_estimates() scheduled task checks
the rolling monthly total and fires an alert if MONTHLY_TOKEN_BUDGET_USD is
exceeded.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AICallType(str, enum.Enum):
    INITIAL_PLAN = "initial_plan"
    ADAPT_PLAN = "adapt_plan"
    RED_FLAG = "red_flag"
    FEEDBACK = "feedback"


class TokenUsage(Base):
    """
    Intentionally does not inherit BaseModel (no updated_at — rows are
    append-only).  Uses a simple auto-increment PK for high-throughput inserts.
    """

    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Context ────────────────────────────────────────────────────────────────
    call_type: Mapped[AICallType] = mapped_column(
        Enum(AICallType, name="ai_call_type"),
        nullable=False,
        index=True,
    )
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="NULL for system-level calls not tied to a specific patient.",
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Stored as plain UUID — no FK to avoid locking session rows on insert.",
    )

    # ── Usage ──────────────────────────────────────────────────────────────────
    model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Exact model string, e.g. 'claude-sonnet-4-20250514'.",
    )
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="input_tokens + output_tokens.",
    )
    cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Estimated cost in USD computed from token counts and model pricing.",
    )

    # ── Outcome ────────────────────────────────────────────────────────────────
    was_cached: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        comment="True if the response was served from the Redis feedback cache (no API call made).",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of retries needed due to validation failures.",
    )
    validation_passed: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
        comment="False if response_parser raised a PlanValidationError on the final attempt.",
    )

    # ── Timing ─────────────────────────────────────────────────────────────────
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Wall-clock time for the API call in milliseconds.",
    )

    # ── Computed helpers ───────────────────────────────────────────────────────
    @property
    def is_feedback_call(self) -> bool:
        return self.call_type == AICallType.FEEDBACK