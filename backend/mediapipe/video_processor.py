"""
mediapipe/video_processor.py

Frame-by-frame landmark extraction from pre-recorded video files.

Used by:
  - Celery ``video_tasks.process_intake_video()``      — baseline ROM extraction
  - Celery ``video_tasks.process_session_recording()`` — session metric extraction

Design
------
``process_video_file()`` is the primary entry point.  It opens the video
with OpenCV, samples frames at a configurable rate, extracts landmarks via the
process-level PoseEstimator singleton, computes joint angles, and returns a
list of per-frame dicts.

Frame sampling
--------------
One frame every ``SAMPLE_EVERY_N_FRAMES`` (default 10) is analysed.
At 30 fps this yields ~3 samples/second — enough for slow physiotherapy
movements while cutting CPU time by 10×.

Memory efficiency
-----------------
Frames are yielded lazily by ``_frame_generator()`` and processed one at a
time.  The results list stores only lightweight landmark dicts, not pixels.

Error handling
--------------
Individual frame errors are logged and skipped so a partial video with some
corrupt frames still produces usable results.  A ``VideoProcessingError`` is
raised only if the video file cannot be opened at all.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import cv2

from app.core.exceptions import VideoProcessingError
from app.core.logging import get_logger

# Intra-package imports — use the package name (mediapipe), not app.mediapipe
from mediapipe.joint_angles import compute_all_joint_angles
from mediapipe.pose_estimator import get_estimator

log = get_logger(__name__)

# Sample one frame every N frames (10 → ~3 fps at 30 fps source)
SAMPLE_EVERY_N_FRAMES: int = 10

# Minimum visibility score to count a landmark as reliable
MIN_FRAME_VISIBILITY: float = 0.5

# Minimum fraction of 33 landmarks that must be visible to include the frame
MIN_VISIBLE_FRACTION: float = 0.7   # at least 23 of 33


# ── Primary entry point ───────────────────────────────────────────────────────

def process_video_file(
    video_path: str,
    sample_every: int = SAMPLE_EVERY_N_FRAMES,
) -> list[dict[str, Any]]:
    """
    Extract per-frame landmark data from a video file.

    Args:
        video_path:   Absolute path to a local video file (mp4, mov, avi, webm).
        sample_every: Process one frame every this many frames.
                      Lower = more accurate but slower.

    Returns:
        List of frame dicts ordered chronologically::

            [
              {
                "frame":         int,              # 0-based source frame index
                "timestamp_ms":  int,              # milliseconds from video start
                "landmarks":     list[dict],       # 33-element MediaPipe landmarks
                "joint_angles":  dict[str, float], # pre-computed angles (deg)
                "visible_count": int,              # landmarks above visibility thr
              },
              ...
            ]

        Returns an empty list if no poses were detected or if the video is
        unreadable without raising (only the open-failure case raises).

    Raises:
        VideoProcessingError: Video file not found or OpenCV cannot open it.
    """
    if not os.path.exists(video_path):
        raise VideoProcessingError(
            f"Video file not found: {video_path}",
            detail={"path": video_path},
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoProcessingError(
            f"OpenCV could not open video: {video_path}",
            detail={"path": video_path},
        )

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(
        "video_processing_started",
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        sample_every=sample_every,
    )

    estimator       = get_estimator()
    results: list   = []
    frames_sampled  = 0
    frames_with_pose = 0

    try:
        for frame_index, frame_bgr, timestamp_ms in _frame_generator(cap, fps, sample_every):
            frames_sampled += 1
            try:
                landmarks = estimator.estimate(frame_bgr)
            except Exception as exc:
                log.warning(
                    "frame_landmark_extraction_failed",
                    frame=frame_index,
                    error=str(exc),
                )
                continue

            if not landmarks:
                continue

            # Visibility quality gate
            visible_count = sum(
                1 for lm in landmarks
                if float(lm.get("visibility", 0.0)) >= MIN_FRAME_VISIBILITY
            )
            if visible_count / len(landmarks) < MIN_VISIBLE_FRACTION:
                continue

            joint_angles = compute_all_joint_angles(landmarks)
            frames_with_pose += 1

            results.append({
                "frame":         frame_index,
                "timestamp_ms":  timestamp_ms,
                "landmarks":     landmarks,
                "joint_angles":  joint_angles,
                "visible_count": visible_count,
            })

    finally:
        cap.release()

    log.info(
        "video_processing_complete",
        path=video_path,
        frames_sampled=frames_sampled,
        frames_with_pose=frames_with_pose,
        result_count=len(results),
    )

    return results


# ── Frame generator ───────────────────────────────────────────────────────────

def _frame_generator(
    cap: cv2.VideoCapture,
    fps: float,
    sample_every: int,
) -> Generator[tuple[int, Any, int], None, None]:
    """
    Lazily yield sampled frames from an open VideoCapture.

    Reads sequentially (faster than ``CAP_PROP_POS_FRAMES`` seeking for each
    sample) and skips non-sampled frames without yielding them.

    Yields:
        (frame_index, frame_bgr_ndarray, timestamp_ms) tuples.
    """
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_every == 0:
            timestamp_ms = int((frame_index / fps) * 1000)
            yield frame_index, frame, timestamp_ms

        frame_index += 1


# ── Post-processing helpers ───────────────────────────────────────────────────

def extract_peak_rom(
    frames: list[dict[str, Any]],
    joints: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Extract the peak (maximum) ROM angle per joint across all frames.

    Args:
        frames:  Output from ``process_video_file()``.
        joints:  Optional whitelist of joint names.  If None, all joints
                 present in the data are included.

    Returns:
        ``{joint_name: {"angle_deg": float, "frame_index": int, "timestamp_ms": int}}``
    """
    peaks: dict[str, dict[str, Any]] = {}

    for frame_data in frames:
        frame_idx    = frame_data["frame"]
        timestamp_ms = frame_data["timestamp_ms"]

        for joint, angle in frame_data.get("joint_angles", {}).items():
            if joints is not None and joint not in joints:
                continue
            if joint not in peaks or angle > peaks[joint]["angle_deg"]:
                peaks[joint] = {
                    "angle_deg":    round(float(angle), 1),
                    "frame_index":  frame_idx,
                    "timestamp_ms": timestamp_ms,
                }

    return peaks


def extract_rom_time_series(
    frames: list[dict[str, Any]],
    joint: str,
) -> list[dict[str, Any]]:
    """
    Extract a chronological angle time-series for one joint.

    Args:
        frames: Output from ``process_video_file()``.
        joint:  Joint name to extract, e.g. ``"left_knee"``.

    Returns:
        List of ``{"timestamp_ms": int, "angle_deg": float}`` dicts.
    """
    series = []
    for frame_data in frames:
        angle = frame_data.get("joint_angles", {}).get(joint)
        if angle is not None:
            series.append({
                "timestamp_ms": frame_data["timestamp_ms"],
                "angle_deg":    round(float(angle), 1),
            })
    return series


def video_duration_seconds(frames: list[dict[str, Any]]) -> float:
    """
    Estimate video duration in seconds from the last frame's timestamp.
    Returns 0.0 if the frame list is empty.
    """
    if not frames:
        return 0.0
    return frames[-1]["timestamp_ms"] / 1000.0