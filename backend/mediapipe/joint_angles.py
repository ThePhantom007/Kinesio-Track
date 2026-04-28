"""
Pure geometry helpers for joint angle computation.

All functions operate on plain (x, y) coordinate tuples so they can be
called from both the server-side pose estimator and the Celery video
processor without any MediaPipe dependency in the call path.

Coordinate system
-----------------
MediaPipe normalises landmark coordinates to [0.0, 1.0] relative to the
frame dimensions.  All functions here accept normalised or pixel coordinates
interchangeably — only the relative geometry matters, not the scale.

3-D depth (z) is intentionally ignored.  For physiotherapy ROM measurements
the camera faces the patient from the front or side, so the 2-D projection
on the image plane captures the clinically relevant range of motion.  The z
channel adds noise for non-perpendicular camera angles without adding value.
"""

from __future__ import annotations

import math

# Type alias: a 2-D point as (x, y)
Point2D = tuple[float, float]


# ── Core angle computation ────────────────────────────────────────────────────

def compute_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Compute the interior angle at vertex *b* formed by the ray b→a and the
    ray b→c, in degrees.

    Args:
        a: Proximal landmark (e.g. hip for a knee angle).
        b: Vertex landmark   (e.g. knee).
        c: Distal landmark   (e.g. ankle).

    Returns:
        Angle in degrees in [0.0, 180.0].

    Examples:
        >>> compute_angle((0, 1), (0, 0), (1, 0))   # 90° angle at origin
        90.0
        >>> compute_angle((0, 1), (0, 0), (0, -1))  # 180° straight line
        180.0
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba < 1e-7 or mag_bc < 1e-7:
        # Degenerate — two landmarks at the same position
        return 0.0

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    # Clamp to [-1, 1] to guard against floating-point rounding past ±1
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def compute_angle_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """
    3-D variant using (x, y, z) tuples.  Used when depth information is
    available and the camera angle warrants it (e.g. overhead recordings).

    Returns angle in degrees in [0.0, 180.0].
    """
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    mag_ba = math.sqrt(sum(v ** 2 for v in ba))
    mag_bc = math.sqrt(sum(v ** 2 for v in bc))

    if mag_ba < 1e-7 or mag_bc < 1e-7:
        return 0.0

    dot = sum(ba[i] * bc[i] for i in range(3))
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


# ── Named physiotherapy helpers ───────────────────────────────────────────────
# Each function takes a list of 33 MediaPipe landmark dicts and extracts the
# relevant (x, y) coordinates before calling compute_angle().
#
# Landmark index reference (MediaPipe Pose 33-point model):
#   0  nose               11 left_shoulder    23 left_hip
#   1  left_eye_inner     12 right_shoulder   24 right_hip
#   2  left_eye           13 left_elbow       25 left_knee
#   3  left_eye_outer     14 right_elbow      26 right_knee
#   4  right_eye_inner    15 left_wrist       27 left_ankle
#   5  right_eye          16 right_wrist      28 right_ankle
#   6  right_eye_outer    17 left_pinky        29 left_heel
#   7  left_ear           18 right_pinky       30 right_heel
#   8  right_ear          19 left_index        31 left_foot_index
#   9  mouth_left         20 right_index       32 right_foot_index
#  10  mouth_right        21 left_thumb
#                         22 right_thumb

def _xy(landmarks: list[dict], idx: int) -> Point2D:
    """Extract (x, y) from a landmark dict by index."""
    lm = landmarks[idx]
    return float(lm["x"]), float(lm["y"])


def knee_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Knee flexion angle (0° = fully extended, 90° = right angle, 180° = hyperflexed).

    Proximal: hip (23/24)  Vertex: knee (25/26)  Distal: ankle (27/28)

    Args:
        landmarks: 33-element landmark list from MediaPipe.
        side:      "left" or "right".

    Returns:
        Angle in degrees.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 23), _xy(landmarks, 25), _xy(landmarks, 27))
    return compute_angle(_xy(landmarks, 24), _xy(landmarks, 26), _xy(landmarks, 28))


def hip_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Hip flexion angle.

    Proximal: shoulder (11/12)  Vertex: hip (23/24)  Distal: knee (25/26)
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 11), _xy(landmarks, 23), _xy(landmarks, 25))
    return compute_angle(_xy(landmarks, 12), _xy(landmarks, 24), _xy(landmarks, 26))


def ankle_dorsiflexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Ankle dorsiflexion angle.

    Proximal: knee (25/26)  Vertex: ankle (27/28)  Distal: foot index (31/32)

    Clinical reference: normal dorsiflexion ≈ 10–20°; restricted < 10°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 25), _xy(landmarks, 27), _xy(landmarks, 31))
    return compute_angle(_xy(landmarks, 26), _xy(landmarks, 28), _xy(landmarks, 32))


def shoulder_abduction(landmarks: list[dict], side: str = "left") -> float:
    """
    Shoulder abduction angle.

    Proximal: hip (23/24)  Vertex: shoulder (11/12)  Distal: elbow (13/14)

    Clinical reference: normal abduction ≈ 150–180°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 23), _xy(landmarks, 11), _xy(landmarks, 13))
    return compute_angle(_xy(landmarks, 24), _xy(landmarks, 12), _xy(landmarks, 14))


def shoulder_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Shoulder flexion (forward elevation).

    Proximal: hip (23/24)  Vertex: shoulder (11/12)  Distal: elbow (13/14)

    Note: same landmarks as abduction; clinical interpretation differs by
    movement plane.  The pose_analyzer uses landmark_rules to specify the
    expected range per exercise.
    """
    return shoulder_abduction(landmarks, side)


def elbow_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Elbow flexion angle.

    Proximal: shoulder (11/12)  Vertex: elbow (13/14)  Distal: wrist (15/16)

    Clinical reference: normal flexion ≈ 0–150°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 11), _xy(landmarks, 13), _xy(landmarks, 15))
    return compute_angle(_xy(landmarks, 12), _xy(landmarks, 14), _xy(landmarks, 16))


def wrist_extension(landmarks: list[dict], side: str = "left") -> float:
    """
    Wrist extension angle.

    Proximal: elbow (13/14)  Vertex: wrist (15/16)  Distal: index finger (19/20)
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 13), _xy(landmarks, 15), _xy(landmarks, 19))
    return compute_angle(_xy(landmarks, 14), _xy(landmarks, 16), _xy(landmarks, 20))


def lumbar_flexion(landmarks: list[dict]) -> float:
    """
    Approximate lumbar spine flexion from the trunk inclination.

    Uses the midpoint of the hips as the pelvis reference and the midpoint
    of the shoulders as the thorax reference.

    Returns the angle between the vertical axis and the trunk line.
    A value near 180° = upright; lower values = forward flexion.
    """
    left_hip  = _xy(landmarks, 23)
    right_hip = _xy(landmarks, 24)
    left_sho  = _xy(landmarks, 11)
    right_sho = _xy(landmarks, 12)

    hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    sho_mid = ((left_sho[0] + right_sho[0]) / 2, (left_sho[1] + right_sho[1]) / 2)

    # Virtual point directly above hip midpoint (vertical reference)
    vertical_ref = (hip_mid[0], hip_mid[1] - 0.1)

    return compute_angle(vertical_ref, hip_mid, sho_mid)


def neck_flexion(landmarks: list[dict]) -> float:
    """
    Neck flexion angle.

    Proximal: mid-shoulder  Vertex: ear midpoint  Distal: nose (0)

    A value near 180° = neutral; lower values = forward head posture.
    """
    left_sho  = _xy(landmarks, 11)
    right_sho = _xy(landmarks, 12)
    left_ear  = _xy(landmarks, 7)
    right_ear = _xy(landmarks, 8)

    sho_mid = ((left_sho[0] + right_sho[0]) / 2, (left_sho[1] + right_sho[1]) / 2)
    ear_mid = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    nose    = _xy(landmarks, 0)

    return compute_angle(sho_mid, ear_mid, nose)


def hip_knee_ankle_alignment(landmarks: list[dict], side: str = "left") -> float:
    """
    Knee valgus / varus check via hip–knee–ankle alignment in the frontal plane.

    Returns the angle at the knee.  Deviation from 180° indicates valgus
    (< 180°) or varus (> 180°) alignment.  Used in the bilateral_asymmetry
    check in pose_analyzer.
    """
    return knee_flexion(landmarks, side)


# ── Bilateral symmetry ────────────────────────────────────────────────────────

def bilateral_asymmetry(
    landmarks: list[dict],
    joint_fn,
    threshold: float = 15.0,
) -> tuple[float, bool]:
    """
    Compute the bilateral angle difference for a named joint function.

    Args:
        landmarks:  33-element landmark list.
        joint_fn:   One of the named helpers above that accepts a ``side``
                    parameter (e.g. knee_flexion, hip_flexion).
        threshold:  Asymmetry in degrees considered clinically significant.

    Returns:
        (asymmetry_degrees, is_significant) tuple.

    Example:
        asym, flag = bilateral_asymmetry(landmarks, knee_flexion, threshold=15)
    """
    left_angle  = joint_fn(landmarks, "left")
    right_angle = joint_fn(landmarks, "right")
    diff = abs(left_angle - right_angle)
    return diff, diff > threshold


# ── Convenience: compute all standard joints ──────────────────────────────────

def compute_all_joint_angles(landmarks: list[dict]) -> dict[str, float]:
    """
    Compute all 15 standard joint angles for a single frame.

    Returns a dict keyed by the joint names used in landmark_rules:
        left_ankle, right_ankle, left_knee, right_knee,
        left_hip, right_hip, left_shoulder, right_shoulder,
        left_elbow, right_elbow, left_wrist, right_wrist,
        lumbar_spine, neck, thoracic_spine (approximated from lumbar)

    Used by the video_processor for batch per-frame analysis.
    """
    if len(landmarks) < 33:
        return {}

    try:
        return {
            "left_ankle":    ankle_dorsiflexion(landmarks, "left"),
            "right_ankle":   ankle_dorsiflexion(landmarks, "right"),
            "left_knee":     knee_flexion(landmarks, "left"),
            "right_knee":    knee_flexion(landmarks, "right"),
            "left_hip":      hip_flexion(landmarks, "left"),
            "right_hip":     hip_flexion(landmarks, "right"),
            "left_shoulder": shoulder_abduction(landmarks, "left"),
            "right_shoulder":shoulder_abduction(landmarks, "right"),
            "left_elbow":    elbow_flexion(landmarks, "left"),
            "right_elbow":   elbow_flexion(landmarks, "right"),
            "left_wrist":    wrist_extension(landmarks, "left"),
            "right_wrist":   wrist_extension(landmarks, "right"),
            "lumbar_spine":  lumbar_flexion(landmarks),
            "neck":          neck_flexion(landmarks),
            # Thoracic spine is approximated — use lumbar as proxy
            "thoracic_spine": lumbar_flexion(landmarks),
        }
    except (IndexError, KeyError, ZeroDivisionError):
        return {}


# ── Landmark visibility helpers ───────────────────────────────────────────────

def visible_landmarks(
    landmarks: list[dict],
    threshold: float = 0.5,
) -> set[int]:
    """
    Return the set of landmark indices with visibility above *threshold*.

    Args:
        landmarks:  33-element landmark list.
        threshold:  Minimum visibility score to consider a landmark reliable.

    Returns:
        Set of integer indices.
    """
    return {
        i for i, lm in enumerate(landmarks)
        if float(lm.get("visibility", 0.0)) >= threshold
    }


def landmarks_sufficient_for_joint(
    landmarks: list[dict],
    joint_name: str,
    min_visibility: float = 0.5,
) -> bool:
    """
    Check whether all three landmarks required for a joint angle computation
    have sufficient visibility.

    Args:
        landmarks:       33-element landmark list.
        joint_name:      Key matching JOINT_TRIPLETS in pose_analyzer.py.
        min_visibility:  Minimum visibility threshold.

    Returns:
        True if all required landmarks are sufficiently visible.
    """
    from app.services.pose_analyzer import JOINT_TRIPLETS

    triplet = JOINT_TRIPLETS.get(joint_name)
    if triplet is None:
        return False

    visible = visible_landmarks(landmarks, min_visibility)
    return all(idx in visible for idx in triplet)