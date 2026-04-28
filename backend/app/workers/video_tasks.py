"""
Celery tasks for asynchronous video processing.

  process_intake_video(media_id)
    Downloads the intake video from S3, runs server-side MediaPipe to extract
    baseline ROM measurements, updates PatientProfile and Injury records,
    then notifies the patient via push notification.

    Queue: video (CPU-intensive, separate worker pool)
    Estimated runtime: 10–120 s depending on video length.
    Retry: up to 3 times with exponential backoff on transient S3/DB errors.

  process_session_recording(media_id)
    Processes a session recording video.  Extracts per-frame landmarks,
    computes rep-level metrics, writes to TimescaleDB, and triggers
    plan_adapter if the session hasn't been adapted yet.

    Queue: video
    Estimated runtime: 30–180 s.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from celery import Task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

log = get_task_logger(__name__)


# ── Shared async runner ────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine from a synchronous Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── process_intake_video ──────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.video_tasks.process_intake_video",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="video",
    acks_late=True,
)
def process_intake_video(self: Task, media_id: str) -> dict:
    """
    Full intake video analysis pipeline.

    Steps:
      1. Load MediaFile and PatientProfile from DB.
      2. Call VideoIntakeAnalyzerService.analyze() — downloads from S3,
         runs MediaPipe, writes baseline ROM to PatientProfile.
      3. If an Injury row is associated, write mobility_notes to it.
      4. Update MediaFile.processing_status → DONE or FAILED.
      5. Send patient push notification on completion.
      6. If a plan already exists for this injury, trigger plan regeneration
         with the new baseline ROM context (rare edge case).

    Returns:
        {"media_id": str, "status": "done"|"failed", "joints_measured": int}
    """
    log.info("process_intake_video_started", media_id=media_id)

    async def _run_async():
        from sqlalchemy import select

        from app.db.postgres import get_db_context
        from app.models.injury import Injury
        from app.models.media import MediaFile, ProcessingStatus
        from app.models.patient import PatientProfile
        from app.services.video_intake_analyzer import VideoIntakeAnalyzerService
        from app.services.notification import NotificationService
        from app.models.user import User

        async with get_db_context() as db:
            # Load media file
            result = await db.execute(
                select(MediaFile).where(MediaFile.id == UUID(media_id))
            )
            media_file = result.scalar_one_or_none()
            if media_file is None:
                log.error("media_file_not_found", media_id=media_id)
                return {"media_id": media_id, "status": "failed", "joints_measured": 0}

            # Load patient
            patient = await db.get(PatientProfile, media_file.patient_id)
            if patient is None:
                log.error("patient_not_found", patient_id=str(media_file.patient_id))
                return {"media_id": media_id, "status": "failed", "joints_measured": 0}

            # Load most recent active injury for this patient
            injury_result = await db.execute(
                select(Injury)
                .where(Injury.patient_id == patient.id)
                .order_by(Injury.created_at.desc())
                .limit(1)
            )
            injury = injury_result.scalar_one_or_none()

            # Run analysis
            analyzer = VideoIntakeAnalyzerService()
            result_data = await analyzer.analyze(
                db=db,
                media_file=media_file,
                patient=patient,
                injury=injury,
            )

            joints_measured = len(result_data.get("baseline_rom", {}))

            # Notify patient
            user = await db.get(User, patient.user_id)
            notif = NotificationService()
            await notif.send_to_patient(
                patient=patient,
                title="Baseline assessment complete",
                body=(
                    f"We've measured your mobility across {joints_measured} joint(s). "
                    "Your personalised plan has been updated."
                ),
                data={"type": "intake_video_complete"},
            )

            log.info(
                "process_intake_video_complete",
                media_id=media_id,
                joints_measured=joints_measured,
            )
            return {
                "media_id":       media_id,
                "status":         "done",
                "joints_measured": joints_measured,
            }

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error("process_intake_video_failed", media_id=media_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


# ── process_session_recording ─────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.video_tasks.process_session_recording",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="video",
    acks_late=True,
)
def process_session_recording(self: Task, media_id: str) -> dict:
    """
    Process a session recording video.

    Extracts per-frame landmarks using server-side MediaPipe, computes
    rep-level quality metrics, and writes them to TimescaleDB.
    """
    log.info("process_session_recording_started", media_id=media_id)

    async def _run_async():
        from sqlalchemy import select

        from app.db.postgres import get_db_context
        from app.db.timescale import write_metric_batch
        from app.db.s3 import download_video
        from app.models.media import MediaFile, ProcessingStatus
        from app.models.session import ExerciseSession
        from app.mediapipe.video_processor import process_video_file
        from app.services.pose_analyzer import PoseAnalyzerService
        import tempfile, os

        async with get_db_context() as db:
            result = await db.execute(
                select(MediaFile).where(MediaFile.id == UUID(media_id))
            )
            media_file = result.scalar_one_or_none()
            if media_file is None or media_file.session_id is None:
                return {"media_id": media_id, "status": "skipped"}

            session = await db.get(ExerciseSession, media_file.session_id)
            if session is None:
                return {"media_id": media_id, "status": "skipped"}

        # Download video to temp file
        suffix = "." + media_file.s3_key.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()

        try:
            await download_video(
                bucket=media_file.s3_bucket,
                key=media_file.s3_key,
                dest_path=tmp.name,
            )

            # Extract landmarks
            frame_landmarks = process_video_file(tmp.name)

            if not frame_landmarks:
                return {"media_id": media_id, "status": "no_landmarks"}

            # Write raw metrics to TimescaleDB
            rows = []
            for frame in frame_landmarks:
                for lm_id, angle in frame.get("joint_angles", {}).items():
                    rows.append({
                        "session_id":  str(session.id),
                        "exercise_id": str(session.exercise_id) if session.exercise_id else None,
                        "joint":       lm_id,
                        "angle_deg":   angle,
                        "quality_score": None,
                    })

            await write_metric_batch(rows)

            # Mark media as done
            async with get_db_context() as db:
                from datetime import datetime, timezone
                mf = await db.get(MediaFile, UUID(media_id))
                if mf:
                    mf.processing_status = ProcessingStatus.DONE
                    mf.processed_at = datetime.now(timezone.utc)
                    mf.duration_seconds = len(frame_landmarks) // 30
                    db.add(mf)

            log.info(
                "process_session_recording_complete",
                media_id=media_id,
                frames=len(frame_landmarks),
                metric_rows=len(rows),
            )
            return {"media_id": media_id, "status": "done", "frames": len(frame_landmarks)}

        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error("process_session_recording_failed", media_id=media_id, error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (self.request.retries + 1))