"""
Injury intake endpoint — the entry point for a new treatment course.

  POST /api/v1/intake

Flow
----
  1. Validate InjuryIntakeRequest.
  2. Load the patient's PatientProfile.
  3. Create an Injury row.
  4. If an intake video s3_key was provided, verify the object exists in S3
     and create a MediaFile row.  The Celery video worker will process it
     asynchronously.
  5. Call exercise_planner.create_plan_from_intake() — this calls Claude and
     writes ExercisePlan + PlanPhase + Exercise rows.
  6. If intake video present, enqueue the Celery video processing task.
  7. Commit the transaction and return InjuryIntakeResponse.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.api.deps import (
    CurrentUser,
    DBSession,
    get_exercise_planner,
    get_patient_profile,
)
from app.core.exceptions import NotFoundError
from app.db.s3 import object_exists
from app.models.injury import Injury
from app.models.media import MediaFile, MediaType, ProcessingStatus
from app.schemas.intake import InjuryIntakeRequest, InjuryIntakeResponse

router = APIRouter(prefix="/intake", tags=["intake"])


@router.post("", response_model=InjuryIntakeResponse, status_code=201)
async def injury_intake(
    body: InjuryIntakeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
    planner=Depends(get_exercise_planner),
):
    """
    Submit a new injury for assessment and receive an exercise plan.

    If ``intake_video_s3_key`` is provided the plan is generated using any
    available baseline ROM measurements.  Video analysis (full MediaPipe run)
    is queued as a background task and the patient receives a push notification
    when it completes.

    Returns ``status="generating"`` if plan generation was queued async,
    ``status="ready"`` if it completed inline.
    """
    # Verify S3 object exists before committing anything
    media_file: MediaFile | None = None
    if body.intake_video_s3_key:
        exists = await object_exists(body.intake_video_s3_key)
        if not exists:
            raise NotFoundError(
                f"Intake video not found in storage: {body.intake_video_s3_key}. "
                "Upload the file first using POST /media/upload-url.",
                detail={"s3_key": body.intake_video_s3_key},
            )

    # Write Injury row
    injury = Injury(
        patient_id=patient.id,
        description=body.description,
        body_part=body.body_part,
        pain_score=body.pain_score,
        intake_video_s3_key=body.intake_video_s3_key,
    )
    db.add(injury)
    await db.flush()

    # Write MediaFile row for the intake video
    if body.intake_video_s3_key:
        from app.core.config import settings
        media_file = MediaFile(
            patient_id=patient.id,
            session_id=None,
            s3_key=body.intake_video_s3_key,
            s3_bucket=settings.S3_BUCKET_NAME,
            media_type=MediaType.INTAKE,
            processing_status=ProcessingStatus.PENDING,
        )
        db.add(media_file)
        await db.flush()

    # Generate the exercise plan (calls Claude)
    response = await planner.create_plan_from_intake(
        db=db,
        patient=patient,
        injury=injury,
    )

    # Queue video analysis if intake video provided
    if media_file:
        from app.workers.video_tasks import process_intake_video
        background_tasks.add_task(
            _enqueue_video_task,
            media_id=str(media_file.id),
        )
        response.video_processing_queued = True

    return response


async def _enqueue_video_task(media_id: str) -> None:
    """Enqueue the Celery video processing task (called via BackgroundTasks)."""
    from app.workers.video_tasks import process_intake_video
    process_intake_video.delay(media_id)