"""
Receives one frame's worth of MediaPipe landmarks and the current exercise's
landmark_rules dict, computes joint angles, and checks them against the
configured acceptable ranges.

Performance contract
--------------------
This function is called on every frame received over the WebSocket.
It must return in < 10ms.  No I/O, no async, no external calls.

Violation accumulation
----------------------
The WebSocket handler in session_ws.py calls analyze_frame() on every frame
but only dispatches a FeedbackMessage after POSE_VIOLATION_FRAME_COUNT
consecutive violations of the same joint.  This smooths out momentary
noise and avoids flooding the patient with messages.

Red-flag check
--------------
After angle analysis, evaluate_red_flags() checks the exercise.red_flags
list.  If a condition is met, the result carries severity="stop" or
"seek_care" and the WebSocket handler routes to red_flag_monitor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.exceptions import InsufficientLandmarksError, PoseAnalysisError
from app.core.logging import get_logger

log = get_logger(__name__)

# ── MediaPipe 33-point landmark index map ─────────────────────────────────────

JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    # (proximal, vertex, distal) — angle is measured at the vertex
    "left_knee":        (23, 25, 27),
    "right_knee":       (24, 26, 28),
    "left_hip":         (11, 23, 25),
    "right_hip":        (12, 24, 26),
    "left_shoulder":    (23, 11, 13),
    "right_shoulder":   (24, 12, 14),
    "left_elbow":       (11, 13, 15),
    "right_elbow":      (12, 14, 16),
    "left_wrist":       (13, 15, 17),
    "right_wrist":      (14, 16, 18),
    "left_ankle":       (25, 27, 31),
    "right_ankle":      (26, 28, 32),
    "neck":             (11, 0, 12),
    "lumbar_spine":     (23, 24, 11),   # hips midpoint → spine
    "thoracic_spine":   (11, 12, 23),
}

BILATERAL_PAIRS: dict[str, tuple[str, str]] = {
    "knee":     ("left_knee",     "right_knee"),
    "hip":      ("left_hip",      "right_hip"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "ankle":    ("left_ankle",    "right_ankle"),
    "elbow":    ("left_elbow",    "right_elbow"),
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class JointViolation:
    joint: str
    actual_angle: float
    min_angle: float
    max_angle: float
    deviation_degrees: float           # signed: negative = below min, positive = above max
    deviation_direction: str           # "flexed" | "extended"
    error_type: str                    # e.g. "knee_valgus", "lumbar_hyperextension"
    severity: str                      # "warning" | "error"
    overlay_landmark_ids: list[int]    # landmark indices to highlight in the UI


@dataclass
class FrameAnalysisResult:
    violations: list[JointViolation] = field(default_factory=list)
    bilateral_asymmetry: dict[str, float] = field(default_factory=dict)
    joint_angles: dict[str, float] = field(default_factory=dict)
    form_score: float = 100.0
    red_flag_triggered: bool = False
    red_flag_condition: str | None = None
    red_flag_severity: str | None = None    # "stop" | "seek_care"

    @property
    def has_violations(self) -> bool:
        return bool(self.violations)

    @property
    def worst_violation(self) -> JointViolation | None:
        if not self.violations:
            return None
        return max(self.violations, key=lambda v: abs(v.deviation_degrees))


# ── Error type derivation ──────────────────────────────────────────────────────

_ERROR_TYPES: dict[str, dict[str, str]] = {
    "left_knee":  {"below_min": "knee_hyperflexion",  "above_max": "knee_insufficient_flexion"},
    "right_knee": {"below_min": "knee_hyperflexion",  "above_max": "knee_insufficient_flexion"},
    "left_hip":   {"below_min": "hip_hyperflexion",   "above_max": "hip_insufficient_flexion"},
    "right_hip":  {"below_min": "hip_hyperflexion",   "above_max": "hip_insufficient_flexion"},
    "left_shoulder":  {"below_min": "shoulder_hyperflexion", "above_max": "shoulder_elevation"},
    "right_shoulder": {"below_min": "shoulder_hyperflexion", "above_max": "shoulder_elevation"},
    "left_elbow":  {"below_min": "elbow_hyperflexion", "above_max": "elbow_hyperextension"},
    "right_elbow": {"below_min": "elbow_hyperflexion", "above_max": "elbow_hyperextension"},
    "left_ankle":  {"below_min": "ankle_hyperflexion", "above_max": "ankle_insufficient_range"},
    "right_ankle": {"below_min": "ankle_hyperflexion", "above_max": "ankle_insufficient_range"},
    "lumbar_spine":    {"below_min": "lumbar_flexion",        "above_max": "lumbar_hyperextension"},
    "thoracic_spine":  {"below_min": "thoracic_flexion",      "above_max": "thoracic_hyperextension"},
    "neck":            {"below_min": "neck_hyperflexion",      "above_max": "neck_hyperextension"},
}


def _derive_error_type(joint: str, direction: str) -> str:
    key = "below_min" if direction == "flexed" else "above_max"
    return _ERROR_TYPES.get(joint, {}).get(key, f"{joint}_{direction}")


# ── Core analyser ─────────────────────────────────────────────────────────────

class PoseAnalyzerService:
    """
    Stateless rules engine.  Create one instance and reuse across requests.
    All methods are synchronous — no I/O permitted.
    """

    def analyze_frame(
        self,
        landmarks: list[dict[str, float]],
        landmark_rules: dict[str, Any],
        red_flag_rules: list[dict[str, Any]] | None = None,
    ) -> FrameAnalysisResult:
        """
        Analyse one frame of MediaPipe landmarks against the exercise rules.

        Args:
            landmarks:       List of 33 landmark dicts with keys x, y, z, visibility.
            landmark_rules:  {joint_name: {min_angle, max_angle, axis, priority}} dict
                             from the Exercise.landmark_rules JSONB column.
            red_flag_rules:  Optional list of red-flag condition dicts from
                             Exercise.red_flags.

        Returns:
            FrameAnalysisResult with violations, scores, and red-flag state.

        Raises:
            InsufficientLandmarksError: Too many low-visibility landmarks.
            PoseAnalysisError:          Unexpected failure in the rules engine.
        """
        if len(landmarks) < 33:
            raise PoseAnalysisError(
                f"Expected 33 landmarks, received {len(landmarks)}.",
                detail={"received": len(landmarks)},
            )

        self._check_visibility(landmarks)

        result = FrameAnalysisResult()

        try:
            # Compute all joint angles needed by this exercise's rules.
            for joint_name, rule in landmark_rules.items():
                priority = rule.get("priority", "primary")
                if priority == "bilateral":
                    continue   # handled separately below

                if joint_name not in JOINT_TRIPLETS:
                    log.warning("unknown_joint_in_rules", joint=joint_name)
                    continue

                angle = self._compute_joint_angle(landmarks, joint_name)
                result.joint_angles[joint_name] = angle

                min_a = rule["min_angle"]
                max_a = rule["max_angle"]

                if angle < min_a:
                    deviation = angle - min_a   # negative
                    direction = "flexed"
                    severity = (
                        "error" if abs(deviation) > settings.POSE_ERROR_THRESHOLD
                        else "warning"
                    )
                    result.violations.append(JointViolation(
                        joint=joint_name,
                        actual_angle=round(angle, 1),
                        min_angle=min_a,
                        max_angle=max_a,
                        deviation_degrees=round(deviation, 1),
                        deviation_direction=direction,
                        error_type=_derive_error_type(joint_name, direction),
                        severity=severity,
                        overlay_landmark_ids=list(JOINT_TRIPLETS[joint_name]),
                    ))
                elif angle > max_a:
                    deviation = angle - max_a   # positive
                    direction = "extended"
                    severity = (
                        "error" if deviation > settings.POSE_ERROR_THRESHOLD
                        else "warning"
                    )
                    result.violations.append(JointViolation(
                        joint=joint_name,
                        actual_angle=round(angle, 1),
                        min_angle=min_a,
                        max_angle=max_a,
                        deviation_degrees=round(deviation, 1),
                        deviation_direction=direction,
                        error_type=_derive_error_type(joint_name, direction),
                        severity=severity,
                        overlay_landmark_ids=list(JOINT_TRIPLETS[joint_name]),
                    ))

            # Bilateral symmetry checks
            for joint_name, rule in landmark_rules.items():
                if rule.get("priority") != "bilateral":
                    continue
                pair_key = joint_name.replace("left_", "").replace("right_", "")
                if pair_key in BILATERAL_PAIRS:
                    asymmetry = self._compute_bilateral_asymmetry(
                        landmarks, *BILATERAL_PAIRS[pair_key]
                    )
                    result.bilateral_asymmetry[pair_key] = round(asymmetry, 1)
                    if asymmetry > 20.0:
                        result.violations.append(JointViolation(
                            joint=pair_key,
                            actual_angle=asymmetry,
                            min_angle=0.0,
                            max_angle=20.0,
                            deviation_degrees=round(asymmetry - 20.0, 1),
                            deviation_direction="asymmetric",
                            error_type="bilateral_asymmetry",
                            severity="error" if asymmetry > 30.0 else "warning",
                            overlay_landmark_ids=list(JOINT_TRIPLETS.get(
                                BILATERAL_PAIRS[pair_key][0], ()
                            )),
                        ))

        except (PoseAnalysisError, InsufficientLandmarksError):
            raise
        except Exception as exc:
            raise PoseAnalysisError(f"Unexpected pose analysis failure: {exc}") from exc

        # Form score: start at 100, deduct per violation weighted by severity
        penalty = sum(
            min(abs(v.deviation_degrees) * (2.0 if v.severity == "error" else 1.0), 25.0)
            for v in result.violations
        )
        result.form_score = max(0.0, round(100.0 - penalty, 1))

        # Red-flag evaluation
        if red_flag_rules:
            rf = self._evaluate_red_flags(red_flag_rules, result)
            if rf:
                result.red_flag_triggered = True
                result.red_flag_condition = rf["condition"]
                result.red_flag_severity  = rf["action"]

        return result

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _compute_joint_angle(
        self,
        landmarks: list[dict[str, float]],
        joint_name: str,
    ) -> float:
        """
        Compute the angle (in degrees) at the vertex landmark.
        Uses the 2-D (x, y) projection — sufficient for physiotherapy ROM checks.
        """
        a_idx, b_idx, c_idx = JOINT_TRIPLETS[joint_name]
        a = (landmarks[a_idx]["x"], landmarks[a_idx]["y"])
        b = (landmarks[b_idx]["x"], landmarks[b_idx]["y"])
        c = (landmarks[c_idx]["x"], landmarks[c_idx]["y"])

        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot      = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba   = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc   = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag_ba < 1e-6 or mag_bc < 1e-6:
            return 0.0

        cosine = dot / (mag_ba * mag_bc)
        cosine = max(-1.0, min(1.0, cosine))   # clamp for numerical safety
        return math.degrees(math.acos(cosine))

    def _compute_bilateral_asymmetry(
        self,
        landmarks: list[dict[str, float]],
        left_joint: str,
        right_joint: str,
    ) -> float:
        """
        Return the absolute angle difference between left and right joints.
        Values > 20° typically indicate a compensatory movement pattern.
        """
        left_angle  = self._compute_joint_angle(landmarks, left_joint)
        right_angle = self._compute_joint_angle(landmarks, right_joint)
        return abs(left_angle - right_angle)

    def _check_visibility(self, landmarks: list[dict[str, float]]) -> None:
        """
        Raise InsufficientLandmarksError if too many critical landmarks are
        below the visibility threshold.  Allows up to 6 low-confidence points
        (some occlusion is acceptable).
        """
        low_visibility = [
            i for i, lm in enumerate(landmarks)
            if lm.get("visibility", 1.0) < settings.MEDIAPIPE_MIN_VISIBILITY
        ]
        if len(low_visibility) > 6:
            raise InsufficientLandmarksError(
                f"{len(low_visibility)} landmarks below visibility threshold "
                f"({settings.MEDIAPIPE_MIN_VISIBILITY}). Frame may be too dark "
                "or the patient is out of frame.",
                detail={"low_visibility_indices": low_visibility},
            )

    # ── Red-flag evaluator ─────────────────────────────────────────────────────

    def _evaluate_red_flags(
        self,
        red_flag_rules: list[dict[str, Any]],
        result: FrameAnalysisResult,
    ) -> dict[str, Any] | None:
        """
        Evaluate red-flag rules against current frame angles and violations.

        Rules use a simple expression language:
            "<joint>.angle < 40"
            "bilateral_asymmetry > 30"
            "form_score < 30"

        Returns the first matching rule dict, or None.
        """
        frame_ctx = {
            "form_score": result.form_score,
            "bilateral_asymmetry": max(result.bilateral_asymmetry.values(), default=0.0),
            **{
                f"{joint}.angle": angle
                for joint, angle in result.joint_angles.items()
            },
        }

        for rule in red_flag_rules:
            condition = rule.get("condition", "")
            try:
                if self._eval_condition(condition, frame_ctx):
                    return rule
            except Exception as exc:
                log.warning("red_flag_eval_error", condition=condition, error=str(exc))
        return None

    @staticmethod
    def _eval_condition(condition: str, ctx: dict[str, float]) -> bool:
        """
        Safely evaluate a simple two-operand condition string.
        Supports: <, >, <=, >=, ==
        E.g. "left_knee.angle < 40" or "form_score < 30"
        """
        for op in ("<=", ">=", "<", ">", "=="):
            if op in condition:
                left_str, right_str = condition.split(op, 1)
                left_key = left_str.strip()
                right_val = float(right_str.strip())
                left_val  = ctx.get(left_key)
                if left_val is None:
                    return False
                return {
                    "<":  left_val <  right_val,
                    ">":  left_val >  right_val,
                    "<=": left_val <= right_val,
                    ">=": left_val >= right_val,
                    "==": left_val == right_val,
                }[op]
        return False