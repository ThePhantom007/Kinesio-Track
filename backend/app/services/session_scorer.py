"""
Aggregates per-frame quality data for a completed session into the scalar
metrics stored on the ExerciseSession row and written to TimescaleDB.

Called by the Celery post_session_analysis task immediately after
session_manager.end_session() returns, before plan_adapter runs.

Metric definitions
------------------
  avg_quality_score     Mean form score (0–100) across all analysed frames.
  completion_pct        Reps completed / reps prescribed (capped at 1.0).
  total_reps_completed  Absolute rep count from the Redis rep counter.
  total_sets_completed  Derived: floor(reps / prescribed_reps).
  peak_rom_degrees      Maximum joint angle recorded during the session.
                        Taken from the target joint(s) of the exercise.

TimescaleDB writes
------------------
Each rep's metrics are written as individual rows to the session_metric
hypertable via db/timescale.py for time-range queries and aggregations.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.timescale import write_metric_batch
from app.models.session import ExerciseSession

log = get_logger(__name__)


@dataclass
class SessionMetrics:
    """Computed metrics for one completed session."""

    avg_quality_score: float | None
    completion_pct: float | None
    total_reps_completed: int
    total_sets_completed: int
    peak_rom_degrees: float | None
    frame_count: int


class SessionScorerService:

    async def compute_and_persist(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
        frame_scores: list[float],
        frame_angles: list[dict[str, float]],
        reps_completed: int,
        prescribed_reps: int,
        prescribed_sets: int,
        target_joints: list[str],
    ) -> SessionMetrics:
        """
        Compute session-level metrics from frame-level data and write them
        to the ExerciseSession row and the TimescaleDB metric hypertable.

        Args:
            db:               Async session (caller owns transaction).
            session:          ExerciseSession ORM object to update.
            frame_scores:     List of per-frame form_score floats (0–100).
            frame_angles:     List of per-frame {joint_name: angle_deg} dicts.
            reps_completed:   Total reps completed (from Redis rep counter).
            prescribed_reps:  Reps prescribed for this exercise.
            prescribed_sets:  Sets prescribed for this exercise.
            target_joints:    Joint names to extract for ROM metrics.

        Returns:
            SessionMetrics dataclass.
        """
        avg_quality = (
            round(statistics.mean(frame_scores), 1)
            if frame_scores else None
        )

        total_prescribed = prescribed_reps * prescribed_sets
        completion = (
            min(reps_completed / total_prescribed, 1.0)
            if total_prescribed > 0 else None
        )

        sets_completed = reps_completed // max(prescribed_reps, 1)

        peak_rom = self._compute_peak_rom(frame_angles, target_joints)

        metrics = SessionMetrics(
            avg_quality_score=avg_quality,
            completion_pct=round(completion, 3) if completion is not None else None,
            total_reps_completed=reps_completed,
            total_sets_completed=sets_completed,
            peak_rom_degrees=peak_rom,
            frame_count=len(frame_scores),
        )

        # Update the ExerciseSession row
        session.avg_quality_score   = metrics.avg_quality_score
        session.completion_pct      = metrics.completion_pct
        session.total_reps_completed = metrics.total_reps_completed
        session.total_sets_completed = metrics.total_sets_completed
        session.peak_rom_degrees    = metrics.peak_rom_degrees
        db.add(session)

        # Write per-rep rows to TimescaleDB
        await self._write_timescale_metrics(
            db,
            session_id=session.id,
            exercise_id=session.exercise_id,
            frame_angles=frame_angles,
            frame_scores=frame_scores,
            target_joints=target_joints,
        )

        log.info(
            "session_scored",
            session_id=str(session.id),
            avg_quality=avg_quality,
            completion_pct=completion,
            reps=reps_completed,
            peak_rom=peak_rom,
        )
        return metrics

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_peak_rom(
        self,
        frame_angles: list[dict[str, float]],
        target_joints: list[str],
    ) -> float | None:
        """
        Find the maximum angle recorded across all target joints and all frames.
        Returns None if no angle data is available.
        """
        peak: float | None = None
        for frame in frame_angles:
            for joint in target_joints:
                angle = frame.get(joint)
                if angle is not None:
                    if peak is None or angle > peak:
                        peak = angle
        return round(peak, 1) if peak is not None else None

    async def _write_timescale_metrics(
        self,
        db: AsyncSession,
        *,
        session_id: UUID,
        exercise_id: UUID | None,
        frame_angles: list[dict[str, float]],
        frame_scores: list[float],
        target_joints: list[str],
    ) -> None:
        """
        Build metric batch rows and write to the TimescaleDB hypertable.
        Samples every 3rd frame to avoid hypertable bloat on high-FPS sessions.
        """
        rows: list[dict[str, Any]] = []
        for i, (angles, score) in enumerate(zip(frame_angles, frame_scores)):
            if i % 3 != 0:
                continue  # sample every 3rd frame
            for joint in target_joints:
                angle = angles.get(joint)
                if angle is not None:
                    rows.append({
                        "session_id":   str(session_id),
                        "exercise_id":  str(exercise_id) if exercise_id else None,
                        "joint":        joint,
                        "angle_deg":    angle,
                        "quality_score": score,
                    })

        if rows:
            await write_metric_batch(rows)
            log.debug(
                "timescale_metrics_written",
                session_id=str(session_id),
                row_count=len(rows),
            )