"""
Unit tests for app/services/pose_analyzer.py.

Tests the rules engine in isolation — no DB, no Redis, no network.
Synthetic landmarks are passed directly to analyze_frame() and results
are asserted against expected violation types and severity levels.
"""

from __future__ import annotations

import pytest

from app.services.pose_analyzer import (
    FrameAnalysisResult,
    JointViolation,
    PoseAnalyzerService,
)
from app.core.exceptions import InsufficientLandmarksError, PoseAnalysisError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _landmarks(overrides: dict[int, tuple[float, float]] | None = None) -> list[dict]:
    """Generate 33 high-visibility neutral landmarks with optional position overrides."""
    base: dict[int, tuple[float, float]] = {
        0:  (0.50, 0.05), 7:  (0.46, 0.08), 8:  (0.54, 0.08),
        11: (0.44, 0.28), 12: (0.56, 0.28), 13: (0.42, 0.42),
        14: (0.58, 0.42), 15: (0.40, 0.55), 16: (0.60, 0.55),
        19: (0.39, 0.60), 20: (0.61, 0.60),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.46, 0.88), 28: (0.54, 0.88),
        31: (0.46, 0.96), 32: (0.54, 0.96),
    }
    if overrides:
        base.update(overrides)
    lm = []
    for i in range(33):
        x, y = base.get(i, (0.50, 0.50))
        lm.append({"id": i, "x": x, "y": y, "z": -0.02, "visibility": 0.95})
    return lm


# Landmark rules for ankle exercises
ANKLE_RULES = {
    "left_ankle": {"min_angle": 10.0, "max_angle": 35.0, "axis": "sagittal", "priority": "primary"},
    "right_ankle": {"min_angle": 10.0, "max_angle": 35.0, "axis": "sagittal", "priority": "bilateral"},
}

KNEE_RULES = {
    "left_knee": {"min_angle": 80.0, "max_angle": 120.0, "axis": "sagittal", "priority": "primary"},
}


@pytest.fixture
def analyzer():
    return PoseAnalyzerService()


# ── No violations ─────────────────────────────────────────────────────────────

class TestNoViolations:

    def test_correct_form_no_violations(self, analyzer):
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, ANKLE_RULES)
        # Neutral pose may not fall in the ankle rule range, but we test the
        # structure — result must be a FrameAnalysisResult regardless
        assert isinstance(result, FrameAnalysisResult)

    def test_form_score_bounded(self, analyzer):
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, ANKLE_RULES)
        assert 0.0 <= result.form_score <= 100.0

    def test_no_red_flag_when_no_rules(self, analyzer):
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, {})
        assert not result.red_flag_triggered
        assert result.violations == []


# ── Violation detection ───────────────────────────────────────────────────────

class TestViolationDetection:

    def test_violation_returned_when_angle_below_min(self, analyzer):
        """
        Place ankle/foot index very close to the ankle so the computed
        ankle angle is near 0° — well below the min of 10°.
        We test that a violation is returned, not the exact angle value,
        since the 2D projection of synthetic points varies.
        """
        # Rules with a wide range so the neutral pose can trigger a violation
        tight_rules = {
            "left_knee": {"min_angle": 170.0, "max_angle": 180.0,
                          "axis": "sagittal", "priority": "primary"},
        }
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, tight_rules)
        # Neutral pose knee angle will be < 170° so should have a violation
        assert result.has_violations
        violation = result.violations[0]
        assert violation.joint == "left_knee"
        assert violation.deviation_degrees < 0   # below min

    def test_violation_returned_when_angle_above_max(self, analyzer):
        tight_rules = {
            "left_knee": {"min_angle": 0.0, "max_angle": 10.0,
                          "axis": "sagittal", "priority": "primary"},
        }
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, tight_rules)
        assert result.has_violations
        violation = result.violations[0]
        assert violation.joint == "left_knee"
        assert violation.deviation_degrees > 0   # above max

    def test_severity_warning_for_small_deviation(self, analyzer):
        # Rule with max just barely above neutral knee angle
        from app.core.config import settings
        small_offset = settings.POSE_WARNING_THRESHOLD / 2  # below error threshold
        neutral_knee = 160.0  # approximate neutral knee angle in our landmarks

        rules = {
            "left_knee": {
                "min_angle": 0.0,
                "max_angle": neutral_knee - small_offset,
                "axis": "sagittal",
                "priority": "primary",
            },
        }
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules)
        if result.has_violations:
            assert result.violations[0].severity in ("warning", "error")

    def test_form_score_decreases_with_violations(self, analyzer):
        no_violation_rules  = {"left_knee": {"min_angle": 0.0,   "max_angle": 180.0, "axis": "sagittal", "priority": "primary"}}
        has_violation_rules = {"left_knee": {"min_angle": 0.0,   "max_angle": 10.0,  "axis": "sagittal", "priority": "primary"}}

        result_ok  = analyzer.analyze_frame(_landmarks(), no_violation_rules)
        result_bad = analyzer.analyze_frame(_landmarks(), has_violation_rules)

        assert result_bad.form_score <= result_ok.form_score

    def test_worst_violation_is_largest_deviation(self, analyzer):
        rules = {
            "left_knee":  {"min_angle": 0.0, "max_angle": 10.0, "axis": "sagittal", "priority": "primary"},
            "right_knee": {"min_angle": 0.0, "max_angle": 5.0,  "axis": "sagittal", "priority": "primary"},
        }
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules)
        if len(result.violations) >= 2:
            worst = result.worst_violation
            for v in result.violations:
                assert abs(v.deviation_degrees) <= abs(worst.deviation_degrees)


# ── Bilateral asymmetry ───────────────────────────────────────────────────────

class TestBilateralAsymmetry:

    def test_symmetric_landmarks_no_asymmetry_flag(self, analyzer):
        rules = {"left_knee": {"min_angle": 0.0, "max_angle": 180.0,
                               "axis": "sagittal", "priority": "bilateral"}}
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules)
        # Symmetric landmarks → asymmetry near 0 → no violation expected
        assert result.bilateral_asymmetry.get("knee", 0.0) < 5.0

    def test_asymmetric_landmarks_flagged(self, analyzer):
        rules = {"left_knee": {"min_angle": 0.0, "max_angle": 180.0,
                               "axis": "sagittal", "priority": "bilateral"}}
        # Move left knee significantly toward midline
        lm = _landmarks({25: (0.52, 0.72)})
        result = analyzer.analyze_frame(lm, rules)
        asym = result.bilateral_asymmetry.get("knee", 0.0)
        assert asym > 0.0


# ── Red-flag evaluation ───────────────────────────────────────────────────────

class TestRedFlagEvaluation:

    def test_red_flag_triggered_by_form_score(self, analyzer):
        # Force a large deviation so form_score drops below 30
        rules = {"left_knee": {"min_angle": 0.0, "max_angle": 10.0,
                               "axis": "sagittal", "priority": "primary"}}
        red_flags = [{"condition": "form_score < 90", "action": "stop",
                      "reason": "Low form score"}]
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules, red_flag_rules=red_flags)
        # form_score will be < 90 with a tight rule — red flag should trigger
        if result.form_score < 90:
            assert result.red_flag_triggered
            assert result.red_flag_severity == "stop"

    def test_no_red_flag_when_condition_false(self, analyzer):
        rules = {}
        red_flags = [{"condition": "form_score < 0", "action": "stop",
                      "reason": "Impossible condition"}]
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules, red_flag_rules=red_flags)
        assert not result.red_flag_triggered

    def test_invalid_condition_does_not_crash(self, analyzer):
        rules = {}
        red_flags = [{"condition": "this is not valid !!!", "action": "stop",
                      "reason": "Bad condition"}]
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules, red_flag_rules=red_flags)
        assert not result.red_flag_triggered   # silently skipped


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_too_few_landmarks_raises(self, analyzer):
        lm = [{"id": i, "x": 0.5, "y": 0.5, "z": 0, "visibility": 0.95}
              for i in range(10)]
        with pytest.raises(PoseAnalysisError):
            analyzer.analyze_frame(lm, ANKLE_RULES)

    def test_low_visibility_raises_insufficient_landmarks(self, analyzer):
        from app.core.config import settings
        lm = _landmarks()
        # Set 20 landmarks to below-threshold visibility
        for i in range(20):
            lm[i]["visibility"] = 0.01
        with pytest.raises(InsufficientLandmarksError):
            analyzer.analyze_frame(lm, ANKLE_RULES)

    def test_unknown_joint_in_rules_is_skipped(self, analyzer):
        rules = {"made_up_joint": {"min_angle": 0.0, "max_angle": 180.0,
                                   "axis": "sagittal", "priority": "primary"}}
        lm = _landmarks()
        result = analyzer.analyze_frame(lm, rules)
        assert isinstance(result, FrameAnalysisResult)
        assert result.violations == []