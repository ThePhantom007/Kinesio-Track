"""Add red_flag_events and token_usage tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-01-01 00:03:00.000000

Note: The clinician_profiles and clinician_patients tables were already
created in 0001 because patient_profiles references clinician_profiles.
This migration is a no-op structurally but is kept in the sequence for
traceability — it's where you'd add any additional clinician-related
columns or indexes in future without mixing them into migration 0001.

If you need to add a new clinician column later, do it here (as a new
migration numbered 0006+) rather than editing existing migrations.
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # All clinician tables were created in 0001.
    # Add any future clinician-specific additions here.
    pass


def downgrade() -> None:
    pass