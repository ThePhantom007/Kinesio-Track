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

import cv2
import mediapipe as mp
import numpy as np

from app.core.config import settings
from app.core.exceptions import PoseAnalysisError
from app.core.logging import get_logger

log = get_logger(__name__)

# ── Model path ────────────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "pose_landmarker.task",
)


# ── Estimator class ───────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Wraps MediaPipe Pose for per-frame landmark extraction.

    Usage::

        estimator = PoseEstimator()
        landmarks = estimator.estimate(frame_bgr)   # numpy BGR array
        # or
        landmarks = estimator.estimate_from_bytes(jpeg_bytes)

    Thread safety: MediaPipe Pose is not thread-safe when called across
    threads.  Create one instance per thread / per Celery worker process.
    """

    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
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

    def estimate(self, frame_bgr: np.ndarray) -> list[dict[str, float]]:
        """
        Extract 33 pose landmarks from a single BGR frame.

        Args:
            frame_bgr: OpenCV BGR numpy array (H × W × 3, uint8).

        Returns:
            List of 33 landmark dicts:
                [{id, x, y, z, visibility}, ...]
            Returns an empty list if no pose is detected.

        Raises:
            PoseAnalysisError: MediaPipe raised an unexpected exception.
        """
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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

    def estimate_from_bytes(self, jpeg_bytes: bytes) -> list[dict[str, float]]:
        """
        Decode a JPEG byte string and extract landmarks.

        Args:
            jpeg_bytes: Raw JPEG bytes (e.g. from base64 decoding a web frame).

        Returns:
            List of 33 landmark dicts, or empty list if decode or detection fails.
        """
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
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
# Created lazily so import doesn't trigger model loading at module import time.

_estimator: PoseEstimator | None = None


def get_estimator() -> PoseEstimator:
    """
    Return the process-level PoseEstimator singleton.
    Initialises on first call.

    Used by the Celery video_processor to avoid re-loading the model for
    every video.
    """
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator()
    return _estimator