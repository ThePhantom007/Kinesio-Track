"""
Server-side MediaPipe pose estimation package.

Exports the primary entry points used by services and Celery workers:
  - PoseEstimator        — per-frame landmark extraction from BGR frames
  - get_estimator()      — process-level singleton accessor
  - process_video_file() — full video landmark extraction pipeline
  - compute_angle()      — core geometry helper
  - compute_all_joint_angles() — all 15 standard joints from one frame

Note: this package is named 'mediapipe' and sits at the project root,
distinct from the 'mediapipe' PyPI package.  Python resolves the local
package first due to sys.path ordering.  If this causes import conflicts
in tests, add the project root to PYTHONPATH before the site-packages path.
"""

from app.mediapipe.joint_angles import (  # noqa: F401
    bilateral_asymmetry,
    compute_all_joint_angles,
    compute_angle,
    compute_angle_3d,
)
from app.mediapipe.pose_estimator import PoseEstimator, get_estimator  # noqa: F401
from app.mediapipe.video_processor import (  # noqa: F401
    extract_peak_rom,
    extract_rom_time_series,
    process_video_file,
)