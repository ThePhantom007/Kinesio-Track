"""Initial schema — users, patients, clinicians, injuries, exercise_plans, plan_phases, exercises

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from alembic import op

# revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:

    # ── ENUM types ─────────────────────────────────────────────────────────────

    op.execute("CREATE TYPE user_role AS ENUM ('patient', 'clinician', 'admin')")
    op.execute("CREATE TYPE activity_level AS ENUM ('sedentary', 'lightly_active', 'moderately_active', 'very_active')")
    op.execute("CREATE TYPE injury_status AS ENUM ('active', 'resolved', 'on_hold')")
    op.execute("CREATE TYPE body_part AS ENUM ('ankle', 'knee', 'hip', 'lower_back', 'upper_back', 'shoulder', 'elbow', 'wrist', 'neck', 'other')")
    op.execute("CREATE TYPE plan_status AS ENUM ('active', 'completed', 'paused', 'superseded')")

    # ── users ──────────────────────────────────────────────────────────────────

    op.create_table(
        "users",
        sa.Column("id",              UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email",           sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(1024), nullable=False),
        sa.Column("role",            sa.Enum("patient", "clinician", "admin", name="user_role"), nullable=False, server_default="patient"),
        sa.Column("is_active",       sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("full_name",       sa.String(256), nullable=True),
        sa.Column("phone",           sa.String(32),  nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",      sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role",  "users", ["role"])

    # ── clinician_profiles ─────────────────────────────────────────────────────

    op.create_table(
        "clinician_profiles",
        sa.Column("id",                    UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",               UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("license_number",        sa.String(128), nullable=False),
        sa.Column("specialty",             sa.String(128), nullable=True),
        sa.Column("institution",           sa.String(256), nullable=True),
        sa.Column("webhook_url",           sa.String(2048), nullable=True),
        sa.Column("email_alerts_enabled",  sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at",            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_clinician_profiles_user_id",        "clinician_profiles", ["user_id"], unique=True)
    op.create_index("ix_clinician_profiles_license_number", "clinician_profiles", ["license_number"], unique=True)

    # ── patient_profiles ───────────────────────────────────────────────────────
    # Note: active_plan_id FK is added after exercise_plans is created (below).

    op.create_table(
        "patient_profiles",
        sa.Column("id",                   UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",              UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("date_of_birth",        sa.Date(), nullable=True),
        sa.Column("gender",               sa.String(32),  nullable=True),
        sa.Column("region",               sa.String(128), nullable=True),
        sa.Column("activity_level",       sa.Enum("sedentary", "lightly_active", "moderately_active", "very_active", name="activity_level"), nullable=True),
        sa.Column("medical_notes",        sa.Text(), nullable=True),
        sa.Column("baseline_rom",         JSONB(), nullable=True),
        sa.Column("mobility_notes",       sa.Text(), nullable=True),
        sa.Column("active_plan_id",       UUID(as_uuid=True), nullable=True),   # FK added below
        sa.Column("assigned_clinician_id",UUID(as_uuid=True), sa.ForeignKey("clinician_profiles.id", ondelete="SET NULL"), nullable=True),
        sa.Column("fcm_token",            sa.String(512), nullable=True),
        sa.Column("web_push_subscription",JSONB(), nullable=True),
        sa.Column("created_at",           sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",           sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_patient_profiles_user_id",              "patient_profiles", ["user_id"], unique=True)
    op.create_index("ix_patient_profiles_assigned_clinician_id","patient_profiles", ["assigned_clinician_id"])

    # ── clinician_patients (M2M join) ──────────────────────────────────────────

    op.create_table(
        "clinician_patients",
        sa.Column("id",            UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("clinician_id",  UUID(as_uuid=True), sa.ForeignKey("clinician_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("patient_id",    UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id",   ondelete="CASCADE"), nullable=False),
        sa.Column("is_active",     sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("assigned_at",   sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("unassigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes",         sa.Text(), nullable=True),
    )
    op.create_unique_constraint("uq_clinician_patient", "clinician_patients", ["clinician_id", "patient_id"])
    op.create_index("ix_clinician_patients_clinician_id", "clinician_patients", ["clinician_id"])
    op.create_index("ix_clinician_patients_patient_id",   "clinician_patients", ["patient_id"])

    # ── injuries ───────────────────────────────────────────────────────────────

    op.create_table(
        "injuries",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id",          UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("description",         sa.Text(), nullable=False),
        sa.Column("body_part",           sa.Enum("ankle","knee","hip","lower_back","upper_back","shoulder","elbow","wrist","neck","other", name="body_part"), nullable=False),
        sa.Column("pain_score",          sa.Integer(), nullable=False),
        sa.Column("status",              sa.Enum("active","resolved","on_hold", name="injury_status"), nullable=False, server_default="active"),
        sa.Column("intake_video_s3_key", sa.String(1024), nullable=True),
        sa.Column("mobility_notes",      sa.Text(), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_injuries_patient_id", "injuries", ["patient_id"])
    op.create_index("ix_injuries_status",     "injuries", ["status"])

    # ── exercise_plans ─────────────────────────────────────────────────────────

    op.create_table(
        "exercise_plans",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id",          UUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("injury_id",           UUID(as_uuid=True), sa.ForeignKey("injuries.id",         ondelete="CASCADE"), nullable=False),
        sa.Column("parent_plan_id",      UUID(as_uuid=True), sa.ForeignKey("exercise_plans.id",   ondelete="SET NULL"), nullable=True),
        sa.Column("title",               sa.String(256), nullable=False),
        sa.Column("version",             sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status",              sa.Enum("active","completed","paused","superseded", name="plan_status"), nullable=False, server_default="active"),
        sa.Column("current_phase",       sa.Integer(), nullable=False, server_default="1"),
        sa.Column("recovery_target_days",sa.Integer(), nullable=True),
        sa.Column("ai_generated",        sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("contraindications",   ARRAY(sa.Text()), nullable=True),
        sa.Column("escalation_criteria", JSONB(), nullable=True),
        sa.Column("clinician_notes",     sa.Text(), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_exercise_plans_patient_id", "exercise_plans", ["patient_id"])
    op.create_index("ix_exercise_plans_injury_id",  "exercise_plans", ["injury_id"])
    op.create_index("ix_exercise_plans_status",     "exercise_plans", ["status"])

    # Now add the deferred FK: patient_profiles.active_plan_id → exercise_plans
    op.create_foreign_key(
        "fk_patient_profiles_active_plan_id",
        "patient_profiles", "exercise_plans",
        ["active_plan_id"], ["id"],
        ondelete="SET NULL",
        use_alter=True,
    )
    op.create_index("ix_patient_profiles_active_plan_id", "patient_profiles", ["active_plan_id"])

    # ── plan_phases ────────────────────────────────────────────────────────────

    op.create_table(
        "plan_phases",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("plan_id",             UUID(as_uuid=True), sa.ForeignKey("exercise_plans.id", ondelete="CASCADE"), nullable=False),
        sa.Column("phase_number",        sa.Integer(), nullable=False),
        sa.Column("name",                sa.String(128), nullable=False),
        sa.Column("goal",                sa.Text(), nullable=False),
        sa.Column("duration_days",       sa.Integer(), nullable=False),
        sa.Column("progression_criteria",sa.Text(), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_plan_phases_plan_id", "plan_phases", ["plan_id"])

    # ── exercises ──────────────────────────────────────────────────────────────

    op.create_table(
        "exercises",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("phase_id",            UUID(as_uuid=True), sa.ForeignKey("plan_phases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("slug",                sa.String(128), nullable=False),
        sa.Column("name",                sa.String(256), nullable=False),
        sa.Column("order_index",         sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sets",                sa.Integer(), nullable=False, server_default="3"),
        sa.Column("reps",                sa.Integer(), nullable=False, server_default="10"),
        sa.Column("hold_seconds",        sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rest_seconds",        sa.Integer(), nullable=False, server_default="30"),
        sa.Column("tempo",               sa.String(32), nullable=True),
        sa.Column("target_joints",       ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("landmark_rules",      JSONB(), nullable=False, server_default="{}"),
        sa.Column("red_flags",           JSONB(), nullable=True),
        sa.Column("patient_instructions",sa.Text(), nullable=True),
        sa.Column("reference_video_url", sa.String(2048), nullable=True),
        sa.Column("difficulty",          sa.String(32), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_exercises_phase_id", "exercises", ["phase_id"])


def downgrade() -> None:
    op.drop_table("exercises")
    op.drop_table("plan_phases")
    op.drop_constraint("fk_patient_profiles_active_plan_id", "patient_profiles", type_="foreignkey")
    op.drop_index("ix_patient_profiles_active_plan_id", "patient_profiles")
    op.drop_table("exercise_plans")
    op.drop_table("injuries")
    op.drop_table("clinician_patients")
    op.drop_table("patient_profiles")
    op.drop_table("clinician_profiles")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS plan_status")
    op.execute("DROP TYPE IF EXISTS body_part")
    op.execute("DROP TYPE IF EXISTS injury_status")
    op.execute("DROP TYPE IF EXISTS activity_level")
    op.execute("DROP TYPE IF EXISTS user_role")