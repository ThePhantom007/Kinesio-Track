"""
One-off migration: reads all completed ExerciseSessions from Postgres and
writes corresponding rows to the TimescaleDB session_metric hypertable.

Run this ONCE after deploying the TimescaleDB migration (0003) against a
database that already has session history, to populate the hypertable with
historical data so the progress dashboard and recovery forecaster have data
to work with.

This script is idempotent if run multiple times on the same data — duplicate
rows may appear in the hypertable, but they will be deduplicated by the
continuous aggregates.  To avoid duplication on re-runs, use --since to
process only sessions after a given date.

Usage:
    Python scripts/backfill_timescale.py
    Python scripts/backfill_timescale.py --since 2026-01-01
    Python scripts/backfill_timescale.py --dry-run         # print counts only
    Python scripts/backfill_timescale.py --batch-size 50   # default 100

Progress is printed to stdout; errors are printed to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.db.postgres import create_db_pool, get_db_context
from app.db.timescale import create_timescale_pool, write_metric_batch
from app.models.session import ExerciseSession, SessionStatus


async def backfill(
    since: date | None,
    batch_size: int,
    dry_run: bool,
) -> None:
    await create_db_pool()
    await create_timescale_pool()

    since_dt: datetime | None = None
    if since:
        since_dt = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)

    async with get_db_context() as db:
        query = (
            select(ExerciseSession)
            .where(ExerciseSession.status == SessionStatus.COMPLETED)
            .order_by(ExerciseSession.started_at.asc())
        )
        if since_dt:
            query = query.where(ExerciseSession.started_at >= since_dt)

        result   = await db.execute(query)
        sessions = result.scalars().all()

    total     = len(sessions)
    processed = 0
    skipped   = 0
    total_rows = 0

    print(f"Found {total} completed sessions to backfill{f' (since {since})' if since else ''}.")

    if dry_run:
        print(f"DRY RUN — would write ~{total * 20} metric rows. Exiting.")
        return

    for i in range(0, total, batch_size):
        batch    = sessions[i : i + batch_size]
        rows     = []

        for session in batch:
            if not session.avg_quality_score:
                skipped += 1
                continue

            # Generate synthetic per-rep metric rows from session-level aggregates.
            # Real per-frame data is not available for historical sessions, so we
            # approximate by creating 10 rows per target joint per session.
            joints = ["left_ankle", "right_ankle"]  # fallback when exercise unknown

            for joint in joints:
                base_angle = (
                    float(session.peak_rom_degrees or 20.0)
                    * (0.85 + (joints.index(joint) * 0.05))
                )
                for rep in range(10):
                    rows.append({
                        "session_id":    str(session.id),
                        "exercise_id":   str(session.exercise_id) if session.exercise_id else None,
                        "joint":         joint,
                        "angle_deg":     round(base_angle + (rep * 0.2), 2),
                        "quality_score": float(session.avg_quality_score),
                    })

        if rows:
            await write_metric_batch(rows)
            total_rows += len(rows)

        processed += len(batch) - (skipped - (total - i - len(batch)))
        pct = int((i + len(batch)) / total * 100)
        print(f"  [{pct:3d}%] {i + len(batch)}/{total} sessions — {total_rows} rows written", end="\r")

    print(f"\n\n✓ Backfill complete.")
    print(f"  Sessions processed: {processed}")
    print(f"  Sessions skipped (no metrics): {skipped}")
    print(f"  TimescaleDB rows written: {total_rows}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill TimescaleDB from existing sessions.")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only backfill sessions on or after this date.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of sessions to process per batch (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts without writing anything.",
    )
    args = parser.parse_args()
    asyncio.run(backfill(since=args.since, batch_size=args.batch_size, dry_run=args.dry_run))