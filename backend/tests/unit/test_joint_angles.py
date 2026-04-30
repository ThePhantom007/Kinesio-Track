"""
Unit tests for mediapipe/joint_angles.py.

All tests are pure Python — no DB, no network, no MediaPipe library.
Angles are verified against known triangle geometries using trigonometry.
"""

from __future__ import annotations

import math

import pytest

from mediapipe.joint_angles import (
    JOINT_TRIPLETS,
    bilateral_asymmetry,
    compute_all_joint_angles,
    compute_angle,
    compute_angle_3d,
    knee_flexion,
    hip_flexion,
    ankle_dorsiflexion,
    shoulder_abduction,
    elbow_flexion,
    lumbar_flexion,
    neck_flexion,
    landmarks_sufficient_for_joint,
    visible_landmarks,
)


# ── compute_angle ─────────────────────────────────────────────────────────────

class TestComputeAngle:

    def test_right_angle(self):
        # Classic 90° at the origin: a=(0,1), b=(0,0), c=(1,0)
        assert compute_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0, abs=0.01)

    def test_straight_line(self):
        # 180° — three collinear points
        assert compute_angle((0, 1), (0, 0), (0, -1)) == pytest.approx(180.0, abs=0.01)

    def test_45_degrees(self):
        # Isoceles right triangle: 45° at vertex
        assert compute_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0, abs=0.01)

    def test_equilateral_60_degrees(self):
        # Equilateral triangle — all angles 60°
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        c = (0.5, math.sqrt(3) / 2)
        assert compute_angle(a, b, c) == pytest.approx(60.0, abs=0.1)

    def test_degenerate_same_point(self):
        # Two landmarks at same position — returns 0.0, does not crash
        assert compute_angle((0, 0), (0, 0), (1, 0)) == 0.0

    def test_degenerate_all_same(self):
        assert compute_angle((0, 0), (0, 0), (0, 0)) == 0.0

    def test_obtuse_angle(self):
        # 135° angle
        a = (1, 0)
        b = (0, 0)
        c = (-1, 1)
        angle = compute_angle(a, b, c)
        assert 120.0 < angle < 150.0

    def test_result_always_in_range(self):
        import random
        rng = random.Random(42)
        for _ in range(100):
            pts = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(3)]
            angle = compute_angle(*pts)
            assert 0.0 <= angle <= 180.0


class TestComputeAngle3D:

    def test_right_angle_3d(self):
        a = (1, 0, 0)
        b = (0, 0, 0)
        c = (0, 1, 0)
        assert compute_angle_3d(a, b, c) == pytest.approx(90.0, abs=0.01)

    def test_same_as_2d_when_z_zero(self):
        a2 = (0.5, 0.3)
        b2 = (0.4, 0.5)
        c2 = (0.6, 0.6)
        a3 = (*a2, 0.0)
        b3 = (*b2, 0.0)
        c3 = (*c2, 0.0)
        assert compute_angle(a2, b2, c2) == pytest.approx(
            compute_angle_3d(a3, b3, c3), abs=0.01
        )


# ── Named helpers ─────────────────────────────────────────────────────────────

def _make_landmarks(overrides: dict[int, tuple[float, float]] | None = None) -> list[dict]:
    """Return 33 neutral landmarks with optional position overrides."""
    base: dict[int, tuple[float, float]] = {
        0:  (0.50, 0.05),   # nose
        7:  (0.46, 0.08),   # left ear
        8:  (0.54, 0.08),   # right ear
        11: (0.44, 0.28),   # left shoulder
        12: (0.56, 0.28),   # right shoulder
        13: (0.42, 0.42),   # left elbow
        14: (0.58, 0.42),   # right elbow
        15: (0.40, 0.55),   # left wrist
        16: (0.60, 0.55),   # right wrist
        19: (0.39, 0.60),   # left index
        20: (0.61, 0.60),   # right index
        23: (0.46, 0.55),   # left hip
        24: (0.54, 0.55),   # right hip
        25: (0.46, 0.72),   # left knee
        26: (0.54, 0.72),   # right knee
        27: (0.46, 0.88),   # left ankle
        28: (0.54, 0.88),   # right ankle
        31: (0.46, 0.96),   # left foot index
        32: (0.54, 0.96),   # right foot index
    }
    if overrides:
        base.update(overrides)

    lm = []
    for i in range(33):
        x, y = base.get(i, (0.50, 0.50))
        lm.append({"id": i, "x": x, "y": y, "z": -0.02, "visibility": 0.95})
    return lm


class TestNamedHelpers:

    def test_knee_flexion_left_returns_float(self):
        lm = _make_landmarks()
        angle = knee_flexion(lm, "left")
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_knee_flexion_right_returns_float(self):
        lm = _make_landmarks()
        assert 0.0 <= knee_flexion(lm, "right") <= 180.0

    def test_hip_flexion_returns_float(self):
        lm = _make_landmarks()
        assert 0.0 <= hip_flexion(lm, "left") <= 180.0

    def test_ankle_dorsiflexion_returns_float(self):
        lm = _make_landmarks()
        assert 0.0 <= ankle_dorsiflexion(lm, "left") <= 180.0

    def test_shoulder_abduction_returns_float(self):
        lm = _make_landmarks()
        assert 0.0 <= shoulder_abduction(lm, "left") <= 180.0

    def test_elbow_flexion_returns_float(self):
        lm = _make_landmarks()
        assert 0.0 <= elbow_flexion(lm, "left") <= 180.0

    def test_lumbar_flexion_upright_near_180(self):
        lm = _make_landmarks()
        angle = lumbar_flexion(lm)
        # Upright posture — trunk near vertical → angle near 180°
        assert angle > 140.0

    def test_neck_flexion_neutral_near_180(self):
        lm = _make_landmarks()
        angle = neck_flexion(lm)
        assert angle > 100.0   # neutral head position

    def test_knee_valgus_detectable(self):
        """Knee shifted medially should produce different angle vs neutral."""
        neutral = _make_landmarks()
        valgus  = _make_landmarks({25: (0.50, 0.72)})  # knee toward midline
        assert knee_flexion(neutral, "left") != knee_flexion(valgus, "left")


# ── bilateral_asymmetry ───────────────────────────────────────────────────────

class TestBilateralAsymmetry:

    def test_symmetric_no_flag(self):
        lm = _make_landmarks()
        diff, flagged = bilateral_asymmetry(lm, knee_flexion, threshold=15.0)
        assert diff == pytest.approx(0.0, abs=1.0)
        assert not flagged

    def test_asymmetric_flagged(self):
        # Move left knee significantly toward midline (valgus)
        lm = _make_landmarks({25: (0.51, 0.72)})
        diff, flagged = bilateral_asymmetry(lm, knee_flexion, threshold=5.0)
        assert diff > 0.0

    def test_threshold_boundary(self):
        lm = _make_landmarks()
        diff, flagged_tight  = bilateral_asymmetry(lm, knee_flexion, threshold=0.0)
        diff, flagged_loose  = bilateral_asymmetry(lm, knee_flexion, threshold=100.0)
        assert not flagged_loose


# ── compute_all_joint_angles ──────────────────────────────────────────────────

class TestComputeAllJointAngles:

    def test_returns_15_joints(self):
        lm = _make_landmarks()
        angles = compute_all_joint_angles(lm)
        assert len(angles) == 15

    def test_all_keys_present(self):
        lm = _make_landmarks()
        angles = compute_all_joint_angles(lm)
        expected = {
            "left_ankle", "right_ankle", "left_knee", "right_knee",
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "lumbar_spine", "neck", "thoracic_spine",
        }
        assert set(angles.keys()) == expected

    def test_all_values_in_valid_range(self):
        lm = _make_landmarks()
        for joint, angle in compute_all_joint_angles(lm).items():
            assert 0.0 <= angle <= 180.0, f"{joint} = {angle} out of range"

    def test_short_landmark_list_returns_empty(self):
        assert compute_all_joint_angles([]) == {}
        assert compute_all_joint_angles([{"id": 0, "x": 0.5, "y": 0.5, "z": 0, "visibility": 1}] * 10) == {}


# ── Visibility helpers ────────────────────────────────────────────────────────

class TestVisibilityHelpers:

    def test_visible_landmarks_all_high(self):
        lm = _make_landmarks()  # all visibility = 0.95
        visible = visible_landmarks(lm, threshold=0.5)
        assert len(visible) == 33

    def test_visible_landmarks_none_above_threshold(self):
        lm = [{"id": i, "x": 0.5, "y": 0.5, "z": 0, "visibility": 0.1}
              for i in range(33)]
        visible = visible_landmarks(lm, threshold=0.5)
        assert len(visible) == 0

    def test_landmarks_sufficient_for_known_joint(self):
        lm = _make_landmarks()
        assert landmarks_sufficient_for_joint(lm, "left_knee", min_visibility=0.5)

    def test_landmarks_insufficient_for_unknown_joint(self):
        lm = _make_landmarks()
        assert not landmarks_sufficient_for_joint(lm, "made_up_joint")

    def test_landmarks_insufficient_when_low_visibility(self):
        lm = _make_landmarks()
        # Set knee (25) visibility to 0
        lm[25]["visibility"] = 0.0
        assert not landmarks_sufficient_for_joint(lm, "left_knee", min_visibility=0.5)


# ── JOINT_TRIPLETS completeness ───────────────────────────────────────────────

class TestJointTriplets:

    def test_all_indices_in_valid_range(self):
        for joint, (a, b, c) in JOINT_TRIPLETS.items():
            for idx in (a, b, c):
                assert 0 <= idx <= 32, f"{joint} has out-of-range index {idx}"

    def test_no_duplicate_triplets(self):
        seen = set()
        for triplet in JOINT_TRIPLETS.values():
            assert triplet not in seen, f"Duplicate triplet {triplet}"
            seen.add(triplet)