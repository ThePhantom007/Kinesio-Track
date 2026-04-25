"""
Extracts baseline range-of-motion measurements from a patient's intake video.

Called by the Celery video_tasks worker after the patient uploads their intake
video.  The results are written to PatientProfile.baseline_rom and
Injury.mobility_notes, which are then fed into the initial plan generation
prompt as context for Claude.

Pipeline
--------
  1. Download the video from S3 to a temporary local path.
  2. Run mediapipe/video_processor.py frame-by-frame to extract landmarks.
  3. For each target joint, compute the peak range-of-motion angle across
     all frames (this captures the patient's maximum mobility at intake).
  4. Generate a plain-language mobility_notes summary.
  5. Write results to the DB and update media processing_status.
  6. Clean up the temporary file.

Output schema (PatientProfile.baseline_rom)
-------------------------------------------
{
  "left_ankle":  {"angle_deg": 28.5, "frame_index": 142, "timestamp_ms": 4733},
  "right_ankle": {"angle_deg": 31.2, "frame_index": 156, "timestamp_ms": 5200},
  ...
}
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import VideoDownloadError, VideoProcessingError
from app.core.logging import get_logger
from app.db.s3 import download_video
from app.mediapipe.video_processor import process_video_file
from app.mediapipe.joint_angles import compute_angle
from app.models.media import MediaFile, ProcessingStatus
from app.models.injury import Injury
from app.models.patient import PatientProfile
from app.services.pose_analyzer import JOINT_TRIPLETS

log = get_logger(__name__)

# Joints assessed in the intake video.
# Subset of JOINT_TRIPLETS focused on the most clinically relevant joints.
_INTAKE_JOINTS = [
    "left_ankle",  "right_ankle",
    "left_knee",   "right_knee",
    "left_hip",    "right_hip",
    "left_shoulder", "right_shoulder",
    "lumbar_spine",  "neck",
]


class VideoIntakeAnalyzerService:

    async def analyze(
        self,
        *,
        db: AsyncSession,
        media_file: MediaFile,
        patient: PatientProfile,
        injury: Injury | None = None,
    ) -> dict[str, Any]:
        """
        Full intake video analysis pipeline.

        Updates:
          - media_file.processing_status → PROCESSING → DONE / FAILED
          - patient.baseline_rom
          - patient.mobility_notes
          - injury.mobility_notes (if provided)

        Returns:
            Dict with baseline_rom and mobility_notes (also written to DB).

        Raises:
            VideoDownloadError:   S3 download failed.
            VideoProcessingError: MediaPipe or frame extraction failed.
        """
        log.info(
            "intake_video_analysis_started",
            media_id=str(media_file.id),
            patient_id=str(patient.id),
            s3_key=media_file.s3_key,
        )

        media_file.processing_status = ProcessingStatus.PROCESSING
        db.add(media_file)
        await db.flush()

        tmp_path: str | None = None
        try:
            tmp_path = await self._download(media_file.s3_key, media_file.s3_bucket)
            frame_landmarks = process_video_file(tmp_path)

            if not frame_landmarks:
                raise VideoProcessingError(
                    "No landmark frames extracted from intake video.",
                    detail={"s3_key": media_file.s3_key},
                )

            baseline_rom = self._extract_peak_rom(frame_landmarks)
            mobility_notes = self._generate_mobility_notes(baseline_rom, injury)

            # Update patient profile
            patient.baseline_rom   = baseline_rom
            patient.mobility_notes = mobility_notes
            db.add(patient)

            # Update injury if provided
            if injury:
                injury.mobility_notes = mobility_notes
                db.add(injury)

            # Mark media as done
            media_file.processing_status = ProcessingStatus.DONE
            media_file.processed_at      = datetime.now(timezone.utc)
            media_file.duration_seconds  = self._estimate_duration(frame_landmarks)
            db.add(media_file)

            await db.flush()

            log.info(
                "intake_video_analysis_complete",
                patient_id=str(patient.id),
                joints_measured=len(baseline_rom),
                frames_processed=len(frame_landmarks),
            )
            return {"baseline_rom": baseline_rom, "mobility_notes": mobility_notes}

        except (VideoDownloadError, VideoProcessingError):
            raise
        except Exception as exc:
            raise VideoProcessingError(
                f"Unexpected error during intake video analysis: {exc}",
                detail={"s3_key": media_file.s3_key},
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Always mark failed if we didn't reach DONE
            if media_file.processing_status == ProcessingStatus.PROCESSING:
                media_file.processing_status = ProcessingStatus.FAILED
                db.add(media_file)
                await db.flush()

    # ── Download ───────────────────────────────────────────────────────────────

    async def _download(self, s3_key: str, bucket: str) -> str:
        """Download the video to a temp file and return its path."""
        suffix = "." + s3_key.split(".")[-1] if "." in s3_key else ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        try:
            await download_video(bucket=bucket, key=s3_key, dest_path=tmp.name)
        except Exception as exc:
            raise VideoDownloadError(
                f"Failed to download {s3_key} from S3: {exc}",
                detail={"s3_key": s3_key, "bucket": bucket},
            ) from exc
        return tmp.name

    # ── ROM extraction ─────────────────────────────────────────────────────────

    def _extract_peak_rom(
        self,
        frame_landmarks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        For each intake joint, find the frame where the angle is largest
        (i.e. the patient's maximum active ROM at intake).

        Returns:
            {joint_name: {angle_deg, frame_index}} dict.
        """
        peaks: dict[str, dict[str, Any]] = {}

        for frame_data in frame_landmarks:
            frame_idx = frame_data.get("frame", 0)
            landmarks = frame_data.get("landmarks", [])
            if len(landmarks) < 33:
                continue

            for joint in _INTAKE_JOINTS:
                if joint not in JOINT_TRIPLETS:
                    continue
                a_idx, b_idx, c_idx = JOINT_TRIPLETS[joint]
                try:
                    a = (landmarks[a_idx]["x"], landmarks[a_idx]["y"])
                    b = (landmarks[b_idx]["x"], landmarks[b_idx]["y"])
                    c = (landmarks[c_idx]["x"], landmarks[c_idx]["y"])
                    angle = compute_angle(a, b, c)
                except (IndexError, KeyError):
                    continue

                if joint not in peaks or angle > peaks[joint]["angle_deg"]:
                    peaks[joint] = {
                        "angle_deg":   round(angle, 1),
                        "frame_index": frame_idx,
                    }

        return peaks

    # ── Mobility notes ────────────────────────────────────────────────────────

    def _generate_mobility_notes(
        self,
        baseline_rom: dict[str, Any],
        injury: Injury | None,
    ) -> str:
        """
        Generate a concise plain-language mobility summary from the ROM data.
        This is a rule-based summary — Claude is NOT called here to keep the
        Celery worker fast.  Claude receives this as context during plan gen.
        """
        if not baseline_rom:
            return "Baseline ROM measurements could not be extracted from the intake video."

        lines = ["Baseline ROM at intake:"]
        for joint, data in sorted(baseline_rom.items()):
            angle = data.get("angle_deg", 0)
            label = self._interpret_rom(joint, angle)
            lines.append(f"  {joint.replace('_', ' ').title()}: {angle}° ({label})")

        if injury:
            lines.append(
                f"\nPrimary affected area: {injury.body_part.value.replace('_', ' ')}. "
                f"Reported pain: {injury.pain_score}/10."
            )
        return "\n".join(lines)

    @staticmethod
    def _interpret_rom(joint: str, angle: float) -> str:
        """Return a qualitative label for a ROM angle."""
        # Rough clinical thresholds per joint — conservative for safety
        thresholds: dict[str, tuple[float, float]] = {
            "left_ankle":    (10.0, 20.0),
            "right_ankle":   (10.0, 20.0),
            "left_knee":     (90.0, 120.0),
            "right_knee":    (90.0, 120.0),
            "left_hip":      (60.0, 90.0),
            "right_hip":     (60.0, 90.0),
            "left_shoulder": (90.0, 150.0),
            "right_shoulder":(90.0, 150.0),
            "lumbar_spine":  (150.0, 170.0),
            "neck":          (120.0, 150.0),
        }
        low, high = thresholds.get(joint, (60.0, 120.0))
        if angle < low:
            return "restricted"
        if angle > high:
            return "full range"
        return "moderate restriction"

    @staticmethod
    def _estimate_duration(frame_landmarks: list[dict[str, Any]]) -> int:
        """Estimate video duration in seconds from frame count at ~30 fps."""
        return max(1, len(frame_landmarks) // 30)