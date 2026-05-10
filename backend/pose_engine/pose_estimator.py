"""
MediaPipe Tasks PoseLandmarker wrapper.
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

class PoseEstimator:
    """
    Wraps MediaPipe Tasks PoseLandmarker.
    Instantiate once as a module-level singleton — model loading is expensive.
    """

    def __init__(self) -> None:
        base_options = mp_python.BaseOptions(
            model_asset_path=settings.MEDIAPIPE_MODEL_PATH
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        log.info("pose_estimator_initialised", model=settings.MEDIAPIPE_MODEL_PATH)

    def estimate(self, frame: np.ndarray) -> list[dict]:
        """
        Run pose estimation on a single BGR frame (from OpenCV).

        Returns:
            List of 33 landmark dicts with keys: id, x, y, z, visibility.
            Empty list if no pose detected.
        """
        # MediaPipe Tasks expects RGB
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return []
        return [
            {
                "id":         i,
                "x":          lm.x,
                "y":          lm.y,
                "z":          lm.z,
                "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
            }
            for i, lm in enumerate(result.pose_landmarks[0])
        ]

    def close(self) -> None:
        """Release the landmarker resources."""
        self._landmarker.close()