"""
Pure geometry helpers for joint angle computation.

No external dependencies — only the standard library.
Can be imported safely from anywhere in the project, including from inside
this package, without triggering the mediapipe naming-collision issue.

Coordinate system
-----------------
MediaPipe normalises landmark coordinates to [0.0, 1.0] relative to the
frame dimensions.  All functions here accept normalised or pixel coordinates
interchangeably — only relative geometry matters, not scale.

3-D depth (z) is intentionally ignored for the 2-D helpers.  For
physiotherapy ROM measurements the camera faces the patient from the front
or side, so the 2-D projection captures the clinically relevant range of
motion.  A 3-D variant is provided for overhead recording setups.
"""

from __future__ import annotations

import math

# Type alias: a 2-D point as (x, y)
Point2D = tuple[float, float]


# ── Core angle computation ────────────────────────────────────────────────────

def compute_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Compute the interior angle at vertex *b* formed by the rays b→a and b→c,
    in degrees.

    Args:
        a: Proximal landmark (e.g. hip for a knee angle).
        b: Vertex landmark   (e.g. knee).
        c: Distal landmark   (e.g. ankle).

    Returns:
        Angle in degrees in [0.0, 180.0].

    Examples:
        >>> compute_angle((0, 1), (0, 0), (1, 0))   # 90° right angle
        90.0
        >>> compute_angle((0, 1), (0, 0), (0, -1))  # 180° straight line
        180.0
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba < 1e-7 or mag_bc < 1e-7:
        return 0.0  # degenerate — two landmarks at the same position

    dot    = ba[0] * bc[0] + ba[1] * bc[1]
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))  # clamp for safety
    return math.degrees(math.acos(cosine))


def compute_angle_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """
    3-D variant of compute_angle using (x, y, z) tuples.

    Useful when depth information is meaningful (e.g. overhead recordings or
    when MediaPipe's z channel is reliable for the exercise being assessed).

    Returns angle in degrees in [0.0, 180.0].
    """
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    mag_ba = math.sqrt(sum(v ** 2 for v in ba))
    mag_bc = math.sqrt(sum(v ** 2 for v in bc))

    if mag_ba < 1e-7 or mag_bc < 1e-7:
        return 0.0

    dot    = sum(ba[i] * bc[i] for i in range(3))
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


# ── MediaPipe 33-point landmark index map ─────────────────────────────────────
#
# (proximal, vertex, distal) — angle is measured at the vertex
#
# Index reference:
#   0  nose           11 left_shoulder    23 left_hip
#   7  left_ear       12 right_shoulder   24 right_hip
#   8  right_ear      13 left_elbow       25 left_knee
#                     14 right_elbow      26 right_knee
#                     15 left_wrist       27 left_ankle
#                     16 right_wrist      28 right_ankle
#                     19 left_index       31 left_foot_index
#                     20 right_index      32 right_foot_index

JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_knee":       (23, 25, 27),
    "right_knee":      (24, 26, 28),
    "left_hip":        (11, 23, 25),
    "right_hip":       (12, 24, 26),
    "left_shoulder":   (23, 11, 13),
    "right_shoulder":  (24, 12, 14),
    "left_elbow":      (11, 13, 15),
    "right_elbow":     (12, 14, 16),
    "left_wrist":      (13, 15, 19),
    "right_wrist":     (14, 16, 20),
    "left_ankle":      (25, 27, 31),
    "right_ankle":     (26, 28, 32),
    "neck":            (11, 0, 12),
    "lumbar_spine":    (23, 24, 11),
    "thoracic_spine":  (11, 12, 23),
}


# ── Landmark accessor ─────────────────────────────────────────────────────────

def _xy(landmarks: list[dict], idx: int) -> Point2D:
    """Extract (x, y) from a landmark dict by index."""
    lm = landmarks[idx]
    return (float(lm["x"]), float(lm["y"]))


# ── Named physiotherapy helpers ───────────────────────────────────────────────

def knee_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Knee flexion angle (0° = fully extended, ~90° = right angle).
    Proximal: hip  Vertex: knee  Distal: ankle
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 23), _xy(landmarks, 25), _xy(landmarks, 27))
    return compute_angle(_xy(landmarks, 24), _xy(landmarks, 26), _xy(landmarks, 28))


def hip_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Hip flexion angle.
    Proximal: shoulder  Vertex: hip  Distal: knee
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 11), _xy(landmarks, 23), _xy(landmarks, 25))
    return compute_angle(_xy(landmarks, 12), _xy(landmarks, 24), _xy(landmarks, 26))


def ankle_dorsiflexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Ankle dorsiflexion angle.
    Proximal: knee  Vertex: ankle  Distal: foot index
    Clinical reference: normal dorsiflexion ≈ 10–20°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 25), _xy(landmarks, 27), _xy(landmarks, 31))
    return compute_angle(_xy(landmarks, 26), _xy(landmarks, 28), _xy(landmarks, 32))


def shoulder_abduction(landmarks: list[dict], side: str = "left") -> float:
    """
    Shoulder abduction / flexion angle.
    Proximal: hip  Vertex: shoulder  Distal: elbow
    Clinical reference: normal abduction ≈ 150–180°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 23), _xy(landmarks, 11), _xy(landmarks, 13))
    return compute_angle(_xy(landmarks, 24), _xy(landmarks, 12), _xy(landmarks, 14))


def elbow_flexion(landmarks: list[dict], side: str = "left") -> float:
    """
    Elbow flexion angle.
    Proximal: shoulder  Vertex: elbow  Distal: wrist
    Clinical reference: normal flexion ≈ 0–150°.
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 11), _xy(landmarks, 13), _xy(landmarks, 15))
    return compute_angle(_xy(landmarks, 12), _xy(landmarks, 14), _xy(landmarks, 16))


def wrist_extension(landmarks: list[dict], side: str = "left") -> float:
    """
    Wrist extension angle.
    Proximal: elbow  Vertex: wrist  Distal: index finger
    """
    if side == "left":
        return compute_angle(_xy(landmarks, 13), _xy(landmarks, 15), _xy(landmarks, 19))
    return compute_angle(_xy(landmarks, 14), _xy(landmarks, 16), _xy(landmarks, 20))


def lumbar_flexion(landmarks: list[dict]) -> float:
    """
    Approximate lumbar spine flexion from trunk inclination.
    Uses hip midpoint → shoulder midpoint vs a vertical reference.
    ~180° = upright; lower = forward flexion.
    """
    left_hip = _xy(landmarks, 23)
    right_hip = _xy(landmarks, 24)
    left_sho = _xy(landmarks, 11)
    right_sho = _xy(landmarks, 12)

    hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    sho_mid = ((left_sho[0] + right_sho[0]) / 2, (left_sho[1] + right_sho[1]) / 2)

    # Vertical reference point directly above hip midpoint
    vertical_ref = (hip_mid[0], hip_mid[1] - 0.1)
    return compute_angle(vertical_ref, hip_mid, sho_mid)


def neck_flexion(landmarks: list[dict]) -> float:
    """
    Neck flexion angle via shoulder midpoint → ear midpoint → nose.
    ~180° = neutral; lower = forward head posture.
    """
    left_sho = _xy(landmarks, 11)
    right_sho = _xy(landmarks, 12)
    left_ear = _xy(landmarks, 7)
    right_ear = _xy(landmarks, 8)

    sho_mid = ((left_sho[0] + right_sho[0]) / 2, (left_sho[1] + right_sho[1]) / 2)
    ear_mid = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    nose    = _xy(landmarks, 0)

    return compute_angle(sho_mid, ear_mid, nose)


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
        joint_fn:   Named helper that accepts a ``side`` kwarg
                    (e.g. knee_flexion, hip_flexion).
        threshold:  Asymmetry in degrees considered clinically significant.

    Returns:
        (asymmetry_degrees, is_significant) tuple.

    Example::
        asym, flag = bilateral_asymmetry(landmarks, knee_flexion, threshold=15)
    """
    left_angle  = joint_fn(landmarks, "left")
    right_angle = joint_fn(landmarks, "right")
    diff = abs(left_angle - right_angle)
    return diff, diff > threshold


# ── Batch computation ─────────────────────────────────────────────────────────

def compute_all_joint_angles(landmarks: list[dict]) -> dict[str, float]:
    """
    Compute all 15 standard physiotherapy joint angles for one frame.

    Returns a dict keyed by the joint names used in landmark_rules:
        left_ankle, right_ankle, left_knee, right_knee,
        left_hip, right_hip, left_shoulder, right_shoulder,
        left_elbow, right_elbow, left_wrist, right_wrist,
        lumbar_spine, neck, thoracic_spine

    Returns an empty dict if the landmark list is incomplete or if any
    computation fails — callers must handle the empty case gracefully.
    """
    if len(landmarks) < 33:
        return {}

    try:
        return {
            "left_ankle":     ankle_dorsiflexion(landmarks, "left"),
            "right_ankle":    ankle_dorsiflexion(landmarks, "right"),
            "left_knee":      knee_flexion(landmarks, "left"),
            "right_knee":     knee_flexion(landmarks, "right"),
            "left_hip":       hip_flexion(landmarks, "left"),
            "right_hip":      hip_flexion(landmarks, "right"),
            "left_shoulder":  shoulder_abduction(landmarks, "left"),
            "right_shoulder": shoulder_abduction(landmarks, "right"),
            "left_elbow":     elbow_flexion(landmarks, "left"),
            "right_elbow":    elbow_flexion(landmarks, "right"),
            "left_wrist":     wrist_extension(landmarks, "left"),
            "right_wrist":    wrist_extension(landmarks, "right"),
            "lumbar_spine":   lumbar_flexion(landmarks),
            "neck":           neck_flexion(landmarks),
            # Thoracic spine approximated from trunk inclination
            "thoracic_spine": lumbar_flexion(landmarks),
        }
    except (IndexError, KeyError, ZeroDivisionError):
        return {}


# ── Visibility helpers ────────────────────────────────────────────────────────

def visible_landmarks(landmarks: list[dict], threshold: float = 0.5) -> set[int]:
    """
    Return the set of landmark indices with visibility above *threshold*.
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
    Return True if all three landmarks required for *joint_name* have
    sufficient visibility.  Uses JOINT_TRIPLETS defined in this module
    (not imported from pose_analyzer) to avoid cross-package coupling.
    """
    triplet = JOINT_TRIPLETS.get(joint_name)
    if triplet is None:
        return False
    visible = visible_landmarks(landmarks, min_visibility)
    return all(idx in visible for idx in triplet)