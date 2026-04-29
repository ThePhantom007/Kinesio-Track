"""
Server-side MediaPipe PoseLandmarker wrapper.

Used in two scenarios:
  1. Web browser clients — the browser sends raw JPEG frames over the
     WebSocket; the server extracts landmarks here rather than on-device.
  2. Video processing — the Celery video_processor calls this indirectly
     via process_video_file().

On-device (Android MediaPipe Tasks) vs server-side
---------------------------------------------------
Android clients using MediaPipe Tasks extract landmarks on the device and
send only the 33-point JSON array over the WebSocket.  No frames hit this
class for Android clients.

For web clients, a raw_frame_b64 payload arrives and this class is called
from a thread pool executor (asyncio.run_in_executor) to avoid blocking the
event loop — MediaPipe's Python bindings are synchronous.

Model loading
-------------
The PoseLandmarker model is loaded once when PoseEstimator is instantiated.
For the API process this is at app startup; for Celery workers it is loaded
on first use.

The model file is downloaded at Docker build time by scripts/download_models.sh
and is NOT committed to the repository (listed in .gitignore).

Model path: mediapipe/models/pose_landmarker.task
Model:      Full variant — 33 landmarks, the best accuracy.
Complexity: Controlled by MEDIAPIPE_MODEL_COMPLEXITY in settings (0/1/2).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# ── PyPI mediapipe import — bypass our local mediapipe/ package ───────────────
#
# Remove the project root from sys.path so Python resolves 'import mediapipe'
# to the site-packages installation, not this directory.

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_removed = False
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
    _removed = True

try:
    import mediapipe as _mp_lib          # PyPI mediapipe
finally:
    if _removed and _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)   # always restore

# Aliases so the rest of this file reads cleanly
_mp_solutions = _mp_lib.solutions

# ── App imports (safe now that sys.path is restored) ──────────────────────────

from app.core.config import settings
from app.core.exceptions import PoseAnalysisError
from app.core.logging import get_logger

log = get_logger(__name__)


# ── Estimator ─────────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Wraps ``mediapipe.solutions.pose.Pose`` for per-frame landmark extraction.

    Thread safety: MediaPipe Pose is not thread-safe across threads.
    Create one instance per thread / per Celery worker process.

    Usage::

        estimator = PoseEstimator()
        landmarks = estimator.estimate(frame_bgr)          # numpy BGR array
        landmarks = estimator.estimate_from_bytes(jpeg_bytes)  # raw JPEG bytes
    """

    def __init__(self) -> None:
        self._pose = _mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        log.debug(
            "pose_estimator_initialised",
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
        )

    def estimate(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Extract 33 pose landmarks from a single BGR frame.

        Args:
            frame_bgr: OpenCV BGR numpy array (H × W × 3, uint8).

        Returns:
            List of 33 landmark dicts:
                [{"id": int, "x": float, "y": float, "z": float, "visibility": float}, ...]
            Returns an empty list if no pose is detected.

        Raises:
            PoseAnalysisError: MediaPipe raised an unexpected exception.
        """
        try:
            rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
        except Exception as exc:
            raise PoseAnalysisError(
                f"MediaPipe Pose processing failed: {exc}",
            ) from exc

        if result.pose_landmarks is None:
            return []

        return [
            {
                "id":         i,
                "x":          float(lm.x),
                "y":          float(lm.y),
                "z":          float(lm.z),
                "visibility": float(lm.visibility),
            }
            for i, lm in enumerate(result.pose_landmarks.landmark)
        ]

    def estimate_from_bytes(self, jpeg_bytes: bytes) -> list[dict]:
        """
        Decode a JPEG byte string and extract landmarks.

        Args:
            jpeg_bytes: Raw JPEG bytes (e.g. base64-decoded web frame).

        Returns:
            List of 33 landmark dicts, or empty list on decode/detection failure.
        """
        try:
            arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.warning("frame_decode_failed", error=str(exc))
            return []

        if frame is None:
            log.warning("frame_decode_returned_none")
            return []

        return self.estimate(frame)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        log.debug("pose_estimator_closed")

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Process-level singleton ───────────────────────────────────────────────────

_estimator: PoseEstimator | None = None


def get_estimator() -> PoseEstimator:
    """
    Return the process-level PoseEstimator singleton, creating it on first call.

    Used by the Celery video_processor to avoid reloading the model for
    every video file.  Each Celery worker process gets its own singleton.
    """
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator()
    return _estimator