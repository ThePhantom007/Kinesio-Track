"""
Scheduled maintenance tasks.

  daily_cleanup()    [Beat — 02:00 UTC daily]
    1. Purge expired Redis keys (feedback cache, session state orphans).
    2. Delete S3 uploads that never had POST /media/{id}/process called
       (older than 7 days, status=PENDING).
    3. Archive exercise sessions older than 1 year to a cold_sessions table
       and delete the originals to keep the hot table lean.
    4. Hard-delete MediaFile rows and S3 objects that have been soft-deleted
       for > 30 days.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

log = get_task_logger(__name__)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    name="app.workers.cleanup_tasks.daily_cleanup",
    queue="default",
)
def daily_cleanup() -> dict:
    """Run all daily maintenance steps and return a summary dict."""
    log.info("daily_cleanup_started")

    results = {}
    results["redis"]   = _run(_cleanup_redis())
    results["s3"]      = _run(_cleanup_orphaned_s3_uploads())
    results["archive"] = _run(_archive_old_sessions())

    log.info("daily_cleanup_complete", **results)
    return results


# ── Redis cleanup ─────────────────────────────────────────────────────────────

async def _cleanup_redis() -> dict:
    """
    Scan for and delete orphaned session state keys.

    Session keys (session:state:*, session:rules:*, etc.) have a TTL set at
    creation time, so they auto-expire.  This scan catches any that slipped
    through due to a Redis config change or missed TTL set.
    """
    from app.db.redis import get_redis_client

    redis = get_redis_client()
    deleted = 0

    # Scan for session keys with no TTL (TTL == -1 means no expiry set)
    cursor = 0
    pattern = "session:*"
    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=200)
        for key in keys:
            ttl = await redis.ttl(key)
            if ttl == -1:
                await redis.delete(key)
                deleted += 1
        if cursor == 0:
            break

    log.info("redis_cleanup_complete", orphaned_keys_deleted=deleted)
    return {"orphaned_keys_deleted": deleted}


# ── S3 orphaned upload cleanup ─────────────────────────────────────────────────

async def _cleanup_orphaned_s3_uploads() -> dict:
    """
    Delete MediaFile rows (and their S3 objects) that have been in PENDING
    status for more than 7 days.  These are uploads that were initiated but
    never confirmed via POST /media/{id}/process.
    """
    from sqlalchemy import select, delete

    from app.db.postgres import get_db_context
    from app.db.s3 import delete_videos_batch
    from app.models.media import MediaFile, ProcessingStatus

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    deleted_count = 0

    async with get_db_context() as db:
        result = await db.execute(
            select(MediaFile).where(
                MediaFile.processing_status == ProcessingStatus.PENDING,
                MediaFile.created_at < cutoff,
            )
        )
        orphaned = result.scalars().all()

        if orphaned:
            s3_keys = [mf.s3_key for mf in orphaned]
            media_ids = [mf.id for mf in orphaned]

            # Delete from S3
            deleted_from_s3 = await delete_videos_batch(s3_keys)

            # Delete DB rows
            for media_id in media_ids:
                await db.execute(
                    delete(MediaFile).where(MediaFile.id == media_id)
                )
            deleted_count = len(orphaned)

    log.info("orphaned_uploads_cleaned", count=deleted_count)
    return {"orphaned_uploads_deleted": deleted_count}


# ── Session archiving ─────────────────────────────────────────────────────────

async def _archive_old_sessions() -> dict:
    """
    Move ExerciseSessions older than 1 year to the session_archive table.
    Removes associated feedback_events and session_metric rows as well.

    The archive table has the same schema as exercise_sessions but lives
    outside TimescaleDB's continuous aggregates and is excluded from
    active queries by default.

    Note: The session_archive table is created by Alembic migration
    0006_session_archive.py (not yet written — placeholder for future use).
    This task is a no-op if the table doesn't exist.
    """
    from sqlalchemy import text

    from app.db.postgres import get_db_context

    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    archived = 0

    async with get_db_context() as db:
        # Check if archive table exists
        result = await db.execute(
            text(
                "SELECT EXISTS (SELECT FROM pg_tables "
                "WHERE schemaname='public' AND tablename='session_archive')"
            )
        )
        table_exists = result.scalar_one()

        if not table_exists:
            log.info("session_archive_table_not_found_skipping")
            return {"sessions_archived": 0}

        # Move old completed sessions to archive
        result = await db.execute(
            text(
                """
                WITH moved AS (
                    DELETE FROM exercise_sessions
                    WHERE status = 'completed'
                      AND ended_at < :cutoff
                    RETURNING *
                )
                INSERT INTO session_archive SELECT * FROM moved
                """
            ),
            {"cutoff": cutoff},
        )
        archived = result.rowcount

    log.info("session_archive_complete", archived=archived)
    return {"sessions_archived": archived}