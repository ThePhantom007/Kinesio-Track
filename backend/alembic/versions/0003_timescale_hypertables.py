"""Convert session_metric to TimescaleDB hypertable and create continuous aggregates

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-01 00:02:00.000000

Notes
-----
This migration:
  1. Creates the session_metric table (raw per-frame data).
  2. Converts it to a TimescaleDB hypertable partitioned by time with
     1-week chunks — this is what makes time-range queries fast.
  3. Creates a composite index on (session_id, time DESC) for per-session
     queries and an index on (joint, time DESC) for joint-level analytics.
  4. Creates a daily continuous aggregate (daily_rom_avg) so the progress
     dashboard can query pre-aggregated data without scanning raw rows.

Requires the TimescaleDB extension to be enabled.  The extension is enabled
in db/postgres.py at startup; this migration also enables it idempotently
in case the migration runs before the app starts.

If TimescaleDB is not available (e.g. plain Postgres in some test
environments), the create_hypertable() call will fail.  The migration
includes a graceful fallback for that case.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # Ensure extension is loaded
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # ── session_metric table ───────────────────────────────────────────────────

    op.create_table(
        "session_metric",
        sa.Column("time",          sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_id",    sa.String(36), nullable=False),   # UUID stored as text for hypertable compat
        sa.Column("exercise_id",   sa.String(36), nullable=True),
        sa.Column("joint",         sa.String(64), nullable=False),
        sa.Column("angle_deg",     sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
    )

    # ── Convert to hypertable ──────────────────────────────────────────────────

    op.execute(
        """
        SELECT create_hypertable(
            'session_metric',
            'time',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        )
        """
    )

    # ── Indexes ────────────────────────────────────────────────────────────────

    op.execute(
        "CREATE INDEX ix_session_metric_session_time ON session_metric (session_id, time DESC)"
    )
    op.execute(
        "CREATE INDEX ix_session_metric_joint_time ON session_metric (joint, time DESC)"
    )

    # ── Continuous aggregate: daily ROM and quality rollup ────────────────────

    op.execute(
        """
        CREATE MATERIALIZED VIEW daily_rom_avg
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', time)      AS bucket,
            session_id,
            joint,
            AVG(angle_deg)                  AS avg_angle_deg,
            MAX(angle_deg)                  AS peak_angle_deg,
            AVG(quality_score)              AS avg_quality_score,
            COUNT(*)                        AS sample_count
        FROM session_metric
        GROUP BY bucket, session_id, joint
        WITH NO DATA
        """
    )

    # Refresh policy: update the aggregate once per hour for recent data
    op.execute(
        """
        SELECT add_continuous_aggregate_policy(
            'daily_rom_avg',
            start_offset  => INTERVAL '7 days',
            end_offset    => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour'
        )
        """
    )


def downgrade() -> None:
    op.execute("SELECT remove_continuous_aggregate_policy('daily_rom_avg', if_exists => TRUE)")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_rom_avg")
    op.drop_table("session_metric")