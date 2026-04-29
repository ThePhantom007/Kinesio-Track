"""Add red_flag_events and token_usage tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-01-01 00:04:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # ── ENUM types ─────────────────────────────────────────────────────────────

    op.execute("CREATE TYPE red_flag_severity AS ENUM ('warn', 'stop', 'seek_care')")
    op.execute("CREATE TYPE red_flag_trigger  AS ENUM ('pain_spike', 'rom_regression', 'compensation_pattern', 'bilateral_asymmetry', 'exercise_red_flag', 'clinician_manual')")
    op.execute("CREATE TYPE ai_call_type      AS ENUM ('initial_plan', 'adapt_plan', 'red_flag', 'feedback')")

    # ── red_flag_events ────────────────────────────────────────────────────────

    op.create_table(
        "red_flag_events",
        sa.Column("id",                      UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id",              UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id",  ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",              UUID(as_uuid=True), sa.ForeignKey("exercise_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("trigger_type",            sa.Enum("pain_spike","rom_regression","compensation_pattern","bilateral_asymmetry","exercise_red_flag","clinician_manual", name="red_flag_trigger"), nullable=False),
        sa.Column("trigger_context",         JSONB(), nullable=True),
        sa.Column("severity",                sa.Enum("warn","stop","seek_care", name="red_flag_severity"), nullable=False),
        sa.Column("immediate_action",        sa.Text(), nullable=False),
        sa.Column("clinician_note",          sa.Text(), nullable=False),
        sa.Column("session_recommendation",  sa.Text(), nullable=True),
        sa.Column("claude_raw_response",     JSONB(), nullable=True),
        sa.Column("acknowledged_by",         UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("acknowledged_at",         sa.DateTime(timezone=True), nullable=True),
        sa.Column("clinician_response_notes",sa.Text(), nullable=True),
        sa.Column("clinician_notified_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("notification_method",     sa.String(32), nullable=True),
        sa.Column("created_at",              sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",              sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_red_flag_events_patient_id",  "red_flag_events", ["patient_id"])
    op.create_index("ix_red_flag_events_session_id",  "red_flag_events", ["session_id"])
    op.create_index("ix_red_flag_events_severity",    "red_flag_events", ["severity"])
    op.create_index("ix_red_flag_events_trigger_type","red_flag_events", ["trigger_type"])

    # ── token_usage ────────────────────────────────────────────────────────────
    # Append-only cost tracking — no updated_at, integer PK for fast inserts.

    op.create_table(
        "token_usage",
        sa.Column("id",               sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("call_type",        sa.Enum("initial_plan","adapt_plan","red_flag","feedback", name="ai_call_type"), nullable=False),
        sa.Column("patient_id",       UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_id",       sa.String(36), nullable=True),   # plain text — no FK to avoid locking
        sa.Column("model",            sa.String(128), nullable=False),
        sa.Column("input_tokens",     sa.Integer(), nullable=False),
        sa.Column("output_tokens",    sa.Integer(), nullable=False),
        sa.Column("total_tokens",     sa.Integer(), nullable=False),
        sa.Column("cost_usd",         sa.Float(), nullable=True),
        sa.Column("was_cached",       sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("retry_count",      sa.Integer(), nullable=False, server_default="0"),
        sa.Column("validation_passed",sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("called_at",        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("latency_ms",       sa.Integer(), nullable=True),
    )
    op.create_index("ix_token_usage_call_type",  "token_usage", ["call_type"])
    op.create_index("ix_token_usage_patient_id", "token_usage", ["patient_id"])
    op.create_index("ix_token_usage_called_at",  "token_usage", ["called_at"])


def downgrade() -> None:
    op.drop_table("token_usage")
    op.drop_table("red_flag_events")
    op.execute("DROP TYPE IF EXISTS ai_call_type")
    op.execute("DROP TYPE IF EXISTS red_flag_trigger")
    op.execute("DROP TYPE IF EXISTS red_flag_severity")