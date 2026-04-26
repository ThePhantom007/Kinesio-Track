"""
WebSocket endpoint for live exercise feedback:
  WS /ws/session/{session_id}
  WS /ws/session/{session_id}?monitor=true  (clinician monitoring)

Message flow
------------
  Client → Server  LandmarkFrame   every 3–5 frames
  Client → Server  RepComplete     when a rep is done
  Client → Server  Ping            keep-alive

  Server → Client  FeedbackMessage on pose violation (after N consecutive frames)
  Server → Client  MilestoneMessage on rep/set complete
  Server → Client  ExerciseDoneMessage when exercise is complete
  Server → Client  RedFlagMessage  on danger signal
  Server → Client  SessionSummaryMessage on session end
  Server → Client  Pong            keep-alive reply

Violation accumulation
----------------------
A FeedbackMessage is sent only after POSE_VIOLATION_FRAME_COUNT consecutive
frames with the same joint violation.  This smooths out momentary noise and
avoids flooding the patient.  The violation state is tracked in a per-session
in-memory dict (not Redis — too fast for round-trip latency on every frame).

Performance contract
--------------------
The hot path (FRAME_DATA handling) must complete in < 50 ms.
  - pose_analyzer.analyze_frame() is synchronous, < 10 ms
  - feedback_generator.get_feedback() hits Redis cache in < 2 ms on hit
  - DB writes happen only on session state changes, not per frame
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import (
    get_connection_manager,
    get_feedback_generator,
    get_pose_analyzer,
    get_red_flag_monitor,
    get_session_manager,
)
from app.core.config import settings
from app.core.exceptions import AuthenticationError, SessionNotFoundError
from app.core.logging import get_logger
from app.core.security import decode_access_token
from app.db.postgres import get_db_context
from app.schemas.websocket import (
    ErrorMessage,
    ExerciseDoneMessage,
    FeedbackMessage,
    MilestoneMessage,
    OverlayPoint,
    PongMessage,
    RedFlagMessage,
    WSCloseCode,
)

router = APIRouter(tags=["websocket"])
log = get_logger(__name__)

# Number of consecutive violation frames before a FeedbackMessage fires
_VIOLATION_THRESHOLD = settings.POSE_VIOLATION_FRAME_COUNT


@router.websocket("/ws/session/{session_id}")
async def session_websocket(
    session_id: UUID,
    websocket: WebSocket,
    connection_manager=Depends(get_connection_manager),
    pose_analyzer=Depends(get_pose_analyzer),
    feedback_generator=Depends(get_feedback_generator),
    session_manager=Depends(get_session_manager),
    red_flag_monitor_svc=Depends(get_red_flag_monitor),
):
    """
    Core real-time exercise feedback WebSocket.

    Authentication: JWT in Authorization header or ?token= query param.
    The token is validated before the connection is accepted.
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    is_monitor, patient_id = await _authenticate(websocket, session_id)

    # ── Connect ───────────────────────────────────────────────────────────────
    await connection_manager.connect(session_id, websocket, is_monitor=is_monitor)

    # Per-session violation counter: {joint_name: consecutive_frame_count}
    violation_counts: dict[str, int] = defaultdict(int)
    # Track current exercise details loaded once from Redis
    session_state: dict[str, Any] | None = None
    frame_scores: list[float] = []
    frame_angles: list[dict] = []

    log.info(
        "ws_session_opened",
        session_id=str(session_id),
        is_monitor=is_monitor,
    )

    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, ErrorMessage(
                    code="invalid_json",
                    message="Message must be valid JSON.",
                    timestamp_ms=_now_ms(),
                ).model_dump())
                continue

            msg_type = msg.get("type")

            # ── Ping ──────────────────────────────────────────────────────────
            if msg_type == "PING":
                await _send(websocket, PongMessage(
                    timestamp_ms=msg.get("timestamp_ms", _now_ms()),
                    server_time_ms=_now_ms(),
                ).model_dump())
                continue

            # Monitors receive but don't send landmark data
            if is_monitor:
                continue

            # ── Rep complete ──────────────────────────────────────────────────
            if msg_type == "REP_COMPLETE":
                rep_count = await session_manager.increment_rep_count(session_id)
                state = await session_manager.get_session_state(session_id)
                if state:
                    sets = int(state.get("sets", 3))
                    reps = int(state.get("reps", 10))
                    set_num  = ((rep_count - 1) // reps) + 1
                    rep_in_set = ((rep_count - 1) % reps) + 1

                    milestone = MilestoneMessage(
                        session_id=session_id,
                        timestamp_ms=_now_ms(),
                        milestone_type="rep",
                        rep_number=rep_in_set,
                        set_number=set_num,
                        exercise_id=UUID(state["exercise_id"]),
                        message=f"Rep {rep_in_set} of {reps} — keep it up!",
                    )
                    await connection_manager.send_to_patient(session_id, milestone.model_dump())
                continue

            # ── Frame data ────────────────────────────────────────────────────
            if msg_type != "FRAME_DATA":
                continue

            landmarks = msg.get("landmarks", [])
            timestamp_ms = msg.get("timestamp_ms", _now_ms())
            exercise_id_str = msg.get("exercise_id")

            # Transition to IN_PROGRESS on first frame
            frame_count = await session_manager.increment_frame_count(session_id)
            if frame_count == 1:
                async with get_db_context() as db:
                    await session_manager.mark_session_in_progress(
                        db=db, session_id=session_id
                    )

            # Load session state from Redis (cached — not per-frame DB call)
            if session_state is None or frame_count % 300 == 0:
                session_state = await session_manager.get_session_state(session_id)
                if session_state is None:
                    await websocket.close(code=WSCloseCode.SESSION_NOT_FOUND)
                    return

            # Server-side MediaPipe for web clients (source == "raw_frame")
            if msg.get("source") == "raw_frame" and not landmarks:
                raw_b64 = msg.get("raw_frame_b64")
                if raw_b64:
                    landmarks = await _extract_landmarks_from_frame(raw_b64)

            if not landmarks:
                continue

            # Load landmark rules from Redis
            landmark_rules = await session_manager.get_landmark_rules(session_id)
            red_flag_rules = await session_manager.get_red_flag_rules(session_id)

            if not landmark_rules:
                continue

            # ── Pose analysis (synchronous, < 10 ms) ─────────────────────────
            try:
                analysis = pose_analyzer.analyze_frame(
                    landmarks=landmarks,
                    landmark_rules=landmark_rules,
                    red_flag_rules=red_flag_rules,
                )
            except Exception as exc:
                log.warning("pose_analysis_error", session_id=str(session_id), error=str(exc))
                continue

            # Accumulate metrics for session scoring
            frame_scores.append(analysis.form_score)
            frame_angles.append(analysis.joint_angles)

            # ── Red-flag check ────────────────────────────────────────────────
            if analysis.red_flag_triggered:
                async with get_db_context() as db:
                    from sqlalchemy import select
                    from app.models.session import ExerciseSession
                    from app.models.patient import PatientProfile
                    from app.models.plan import ExercisePlan
                    from app.models.injury import Injury

                    sess = await db.get(ExerciseSession, session_id)
                    patient = await db.get(PatientProfile, sess.patient_id) if sess else None
                    plan = await db.get(ExercisePlan, sess.plan_id) if sess else None

                    injury = None
                    if patient and patient.active_plan_id:
                        from sqlalchemy import select as sa_select
                        inj_result = await db.execute(
                            sa_select(Injury).where(Injury.patient_id == patient.id)
                            .order_by(Injury.created_at.desc()).limit(1)
                        )
                        injury = inj_result.scalar_one_or_none()

                    rf_event = await red_flag_monitor_svc.check_frame_result(
                        db=db,
                        session=sess,
                        analysis=analysis,
                        exercise_name=session_state.get("exercise_name", ""),
                        exercise_slug=session_state.get("exercise_slug", ""),
                        patient=patient,
                        injury=injury,
                        plan=plan,
                    )

                if rf_event:
                    rf_msg = RedFlagMessage(
                        session_id=session_id,
                        timestamp_ms=_now_ms(),
                        severity=rf_event.severity.value,
                        message=rf_event.immediate_action,
                        trigger_type=rf_event.trigger_type.value,
                        red_flag_id=rf_event.id,
                    )
                    await connection_manager.send_to_patient(session_id, rf_msg.model_dump())
                    if rf_event.requires_session_stop:
                        # Enqueue clinician notification
                        from app.workers.notification_tasks import notify_clinician_red_flag
                        notify_clinician_red_flag.delay(str(rf_event.id))
                    continue

            # ── Violation accumulation → FeedbackMessage ──────────────────────
            if analysis.has_violations:
                worst = analysis.worst_violation
                if worst:
                    violation_counts[worst.joint] += 1

                    if violation_counts[worst.joint] >= _VIOLATION_THRESHOLD:
                        # Reset counter so the same joint doesn't fire again
                        # until it's corrected and re-violates
                        violation_counts[worst.joint] = 0

                        # Get feedback message (Redis cache → Claude → fallback)
                        async with get_db_context() as db:
                            message, from_cache = await feedback_generator.get_feedback(
                                violation=worst,
                                exercise_slug=session_state.get("exercise_slug", ""),
                                exercise_name=session_state.get("exercise_name", ""),
                                difficulty=session_state.get("difficulty", "beginner"),
                                db=db,
                                session_id=session_id,
                            )

                        feedback_msg = FeedbackMessage(
                            session_id=session_id,
                            timestamp_ms=timestamp_ms,
                            severity=worst.severity,
                            message=message,
                            affected_joint=worst.joint,
                            error_type=worst.error_type,
                            actual_angle=worst.actual_angle,
                            expected_min_angle=worst.min_angle,
                            expected_max_angle=worst.max_angle,
                            deviation_degrees=worst.deviation_degrees,
                            form_score=analysis.form_score,
                            overlay_points=[
                                OverlayPoint(landmark_id=lid, x=0, y=0, highlight=True)
                                for lid in worst.overlay_landmark_ids
                            ],
                            from_cache=from_cache,
                        )
                        await connection_manager.send_to_patient(
                            session_id, feedback_msg.model_dump()
                        )

                        # Buffer for Redis feedback log
                        await session_manager.append_feedback_event(
                            session_id,
                            {
                                "exercise_id":       session_state.get("exercise_id"),
                                "occurred_at":       datetime.now(timezone.utc).isoformat(),
                                "frame_timestamp_ms": timestamp_ms,
                                "severity":          worst.severity,
                                "error_type":        worst.error_type,
                                "affected_joint":    worst.joint,
                                "actual_angle":      worst.actual_angle,
                                "expected_min_angle": worst.min_angle,
                                "expected_max_angle": worst.max_angle,
                                "deviation_degrees": worst.deviation_degrees,
                                "form_score":        analysis.form_score,
                                "message":           message,
                                "from_cache":        from_cache,
                            },
                        )
            else:
                # No violations — reset all counters for joints that are now clean
                for joint in list(violation_counts.keys()):
                    if joint not in {v.joint for v in analysis.violations}:
                        violation_counts[joint] = 0

    except WebSocketDisconnect:
        log.info("ws_client_disconnected", session_id=str(session_id))
    except Exception as exc:
        log.error("ws_unexpected_error", session_id=str(session_id), error=str(exc))
    finally:
        await connection_manager.disconnect(session_id, websocket, is_monitor=is_monitor)
        log.info(
            "ws_session_closed",
            session_id=str(session_id),
            frames_processed=len(frame_scores),
        )


# ── Auth helper ────────────────────────────────────────────────────────────────

async def _authenticate(
    websocket: WebSocket,
    session_id: UUID,
) -> tuple[bool, UUID | None]:
    """
    Extract and validate the JWT from the WebSocket upgrade request.
    Returns (is_monitor, user_id).
    Closes the socket with AUTH_FAILED if invalid.
    """
    token = (
        websocket.headers.get("Authorization", "").removeprefix("Bearer ")
        or websocket.query_params.get("token", "")
    )

    if not token:
        await websocket.close(code=WSCloseCode.AUTH_FAILED)
        raise AuthenticationError("Missing token.")

    try:
        payload = decode_access_token(token)
    except Exception:
        await websocket.close(code=WSCloseCode.AUTH_FAILED)
        raise AuthenticationError("Invalid token.")

    user_id    = UUID(payload["sub"])
    is_monitor = websocket.query_params.get("monitor", "").lower() == "true"
    return is_monitor, user_id


# ── Utility helpers ────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def _send(websocket: WebSocket, data: dict) -> None:
    """Send a dict as JSON text; swallow send errors silently."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


async def _extract_landmarks_from_frame(raw_b64: str) -> list[dict]:
    """
    Server-side MediaPipe landmark extraction for web browser clients.
    Runs in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    import base64
    import numpy as np

    async def _run():
        try:
            from app.mediapipe.pose_estimator import PoseEstimator
            img_bytes = base64.b64decode(raw_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            import cv2
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return []
            estimator = PoseEstimator()
            return estimator.estimate(frame)
        except Exception as exc:
            log.warning("server_side_mediapipe_failed", error=str(exc))
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: asyncio.run(_run()))