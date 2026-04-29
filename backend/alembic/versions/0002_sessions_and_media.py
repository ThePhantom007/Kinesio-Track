"""Add exercise_sessions, media_files, and feedback_events

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-01 00:01:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # ── ENUM types ─────────────────────────────────────────────────────────────

    op.execute("CREATE TYPE session_status    AS ENUM ('pending', 'in_progress', 'completed', 'abandoned')")
    op.execute("CREATE TYPE media_type        AS ENUM ('intake', 'session_recording')")
    op.execute("CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'done', 'failed')")
    op.execute("CREATE TYPE feedback_severity AS ENUM ('info', 'warning', 'error', 'stop')")

    # ── exercise_sessions ──────────────────────────────────────────────────────

    op.create_table(
        "exercise_sessions",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id",          UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("plan_id",             UUID(as_uuid=True), sa.ForeignKey("exercise_plans.id",   ondelete="CASCADE"), nullable=False),
        sa.Column("exercise_id",         UUID(as_uuid=True), sa.ForeignKey("exercises.id",        ondelete="SET NULL"), nullable=True),
        sa.Column("status",              sa.Enum("pending","in_progress","completed","abandoned", name="session_status"), nullable=False, server_default="pending"),
        sa.Column("started_at",          sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at",            sa.DateTime(timezone=True), nullable=True),
        sa.Column("avg_quality_score",   sa.Float(), nullable=True),
        sa.Column("completion_pct",      sa.Float(), nullable=True),
        sa.Column("post_session_pain",   sa.Integer(), nullable=True),
        sa.Column("total_reps_completed",sa.Integer(), nullable=True),
        sa.Column("total_sets_completed",sa.Integer(), nullable=True),
        sa.Column("peak_rom_degrees",    sa.Float(), nullable=True),
        sa.Column("summary_text",        sa.Text(), nullable=True),
        sa.Column("plan_adapted",        sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("patient_notes",       sa.Text(), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_exercise_sessions_patient_id",  "exercise_sessions", ["patient_id"])
    op.create_index("ix_exercise_sessions_plan_id",     "exercise_sessions", ["plan_id"])
    op.create_index("ix_exercise_sessions_status",      "exercise_sessions", ["status"])
    op.create_index("ix_exercise_sessions_started_at",  "exercise_sessions", ["started_at"])

    # ── media_files ────────────────────────────────────────────────────────────

    op.create_table(
        "media_files",
        sa.Column("id",                UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id",        UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",        UUID(as_uuid=True), sa.ForeignKey("exercise_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("s3_key",            sa.String(1024), nullable=False),
        sa.Column("s3_bucket",         sa.String(256),  nullable=False),
        sa.Column("thumbnail_s3_key",  sa.String(1024), nullable=True),
        sa.Column("media_type",        sa.Enum("intake","session_recording", name="media_type"), nullable=False),
        sa.Column("duration_seconds",  sa.Integer(), nullable=True),
        sa.Column("file_size_bytes",   sa.Integer(), nullable=True),
        sa.Column("mime_type",         sa.String(128), nullable=True),
        sa.Column("processing_status", sa.Enum("pending","processing","done","failed", name="processing_status"), nullable=False, server_default="pending"),
        sa.Column("processed_at",      sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_error",  sa.String(1024), nullable=True),
        sa.Column("created_at",        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_media_files_patient_id",        "media_files", ["patient_id"])
    op.create_index("ix_media_files_session_id",        "media_files", ["session_id"])
    op.create_index("ix_media_files_processing_status", "media_files", ["processing_status"])
    op.create_unique_constraint("uq_media_files_s3_key", "media_files", ["s3_key"])

    # ── feedback_events ────────────────────────────────────────────────────────

    op.create_table(
        "feedback_events",
        sa.Column("id",                  sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id",          UUID(as_uuid=True), sa.ForeignKey("exercise_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("exercise_id",         UUID(as_uuid=True), sa.ForeignKey("exercises.id",          ondelete="SET NULL"), nullable=True),
        sa.Column("occurred_at",         sa.DateTime(timezone=True), nullable=False),
        sa.Column("frame_timestamp_ms",  sa.Integer(), nullable=True),
        sa.Column("severity",            sa.Enum("info","warning","error","stop", name="feedback_severity"), nullable=False),
        sa.Column("error_type",          sa.String(128), nullable=True),
        sa.Column("affected_joint",      sa.String(64),  nullable=True),
        sa.Column("actual_angle",        sa.Float(), nullable=True),
        sa.Column("expected_min_angle",  sa.Float(), nullable=True),
        sa.Column("expected_max_angle",  sa.Float(), nullable=True),
        sa.Column("deviation_degrees",   sa.Float(), nullable=True),
        sa.Column("form_score_at_event", sa.Float(), nullable=True),
        sa.Column("message",             sa.Text(), nullable=False),
        sa.Column("from_cache",          sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("overlay_points",      JSONB(), nullable=True),
    )
    op.create_index("ix_feedback_events_session_id",  "feedback_events", ["session_id"])
    op.create_index("ix_feedback_events_occurred_at", "feedback_events", ["occurred_at"])
    op.create_index("ix_feedback_events_severity",    "feedback_events", ["severity"])


def downgrade() -> None:
    op.drop_table("feedback_events")
    op.drop_table("media_files")
    op.drop_table("exercise_sessions")
    op.execute("DROP TYPE IF EXISTS feedback_severity")
    op.execute("DROP TYPE IF EXISTS processing_status")
    op.execute("DROP TYPE IF EXISTS media_type")
    op.execute("DROP TYPE IF EXISTS session_status")