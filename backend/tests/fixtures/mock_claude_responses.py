"""
Hardcoded valid Claude JSON responses for all four call types.
Used to test downstream logic (DB writes, schema validation, plan adaptation)
without making real API calls.

Each constant is the validated Pydantic model instance that claude_client
returns AFTER response_parser has validated the raw JSON.  Tests that need
the raw JSON string should call .model_dump() on these.
"""

from __future__ import annotations

from app.schemas.plan import ExercisePlanAIOutput, PlanPhaseAIOutput, ExerciseAIOutput


# ── Initial plan ──────────────────────────────────────────────────────────────

VALID_PLAN_RESPONSE = ExercisePlanAIOutput(
    title="Ankle Sprain Recovery Programme",
    summary=(
        "A 6-week progressive rehabilitation programme targeting left ankle "
        "stability and range of motion following a Grade II lateral sprain. "
        "Starts with mobility restoration and advances to proprioceptive training."
    ),
    estimated_weeks=6,
    recovery_target_days=42,
    contraindications=[
        "Avoid full weight-bearing jumps until Phase 3",
        "No deep squats below 90° knee flexion",
    ],
    escalation_criteria=[
        {"trigger": "pain_score >= 8", "action": "stop", "reason": "Acute pain spike"},
        {"trigger": "bilateral_asymmetry > 25", "action": "warn", "reason": "Compensation pattern"},
    ],
    phases=[
        PlanPhaseAIOutput(
            phase_number=1,
            name="Phase 1 – Acute Recovery",
            goal="Reduce swelling, restore baseline range of motion to 20°+",
            duration_days=14,
            progression_criteria="avg_quality_score >= 70 over 3 consecutive sessions AND pain <= 5",
            exercises=[
                ExerciseAIOutput(
                    slug="seated-ankle-circles",
                    name="Seated Ankle Circles",
                    sets=3, reps=10, hold_seconds=0, rest_seconds=30,
                    target_joints=["left_ankle", "right_ankle"],
                    landmark_rules={
                        "left_ankle": {"min_angle": 10.0, "max_angle": 35.0,
                                       "axis": "sagittal", "priority": "primary"},
                        "right_ankle": {"min_angle": 10.0, "max_angle": 35.0,
                                        "axis": "sagittal", "priority": "bilateral"},
                    },
                    red_flags=[
                        {"condition": "left_ankle.angle < 5",
                         "action": "stop", "reason": "Hyperflexion risk"},
                    ],
                    patient_instructions=(
                        "Sit in a chair. Lift your foot slightly and rotate "
                        "your ankle slowly in full circles — 10 clockwise, "
                        "10 counter-clockwise."
                    ),
                    difficulty="beginner",
                    safety_warnings=["Stop if you feel sharp pain"],
                ),
                ExerciseAIOutput(
                    slug="towel-toe-curls",
                    name="Towel Toe Curls",
                    sets=2, reps=15, hold_seconds=1, rest_seconds=20,
                    target_joints=["left_ankle"],
                    landmark_rules={
                        "left_ankle": {"min_angle": 15.0, "max_angle": 40.0,
                                       "axis": "sagittal", "priority": "primary"},
                    },
                    red_flags=[],
                    patient_instructions=(
                        "Place a small towel on the floor. Use your toes to "
                        "scrunch and pull it toward you."
                    ),
                    difficulty="beginner",
                    safety_warnings=[],
                ),
            ],
        ),
        PlanPhaseAIOutput(
            phase_number=2,
            name="Phase 2 – Strength Building",
            goal="Rebuild ankle strength and proprioception",
            duration_days=14,
            progression_criteria="avg_quality_score >= 78 over 3 consecutive sessions AND pain <= 3",
            exercises=[
                ExerciseAIOutput(
                    slug="calf-raises",
                    name="Standing Calf Raises",
                    sets=3, reps=15, hold_seconds=2, rest_seconds=45,
                    target_joints=["left_ankle", "right_ankle", "left_knee", "right_knee"],
                    landmark_rules={
                        "left_ankle": {"min_angle": 20.0, "max_angle": 50.0,
                                       "axis": "sagittal", "priority": "primary"},
                        "right_ankle": {"min_angle": 20.0, "max_angle": 50.0,
                                        "axis": "sagittal", "priority": "bilateral"},
                    },
                    red_flags=[],
                    patient_instructions=(
                        "Stand near a wall. Rise onto your toes slowly, "
                        "hold 2 seconds, lower with control."
                    ),
                    difficulty="intermediate",
                    safety_warnings=["Use wall for balance if needed"],
                ),
            ],
        ),
        PlanPhaseAIOutput(
            phase_number=3,
            name="Phase 3 – Return to Function",
            goal="Restore full functional mobility and stability",
            duration_days=14,
            progression_criteria="avg_quality_score >= 85 AND pain <= 2",
            exercises=[
                ExerciseAIOutput(
                    slug="single-leg-balance",
                    name="Single-Leg Balance",
                    sets=3, reps=1, hold_seconds=30, rest_seconds=60,
                    target_joints=["left_ankle", "left_knee", "left_hip"],
                    landmark_rules={
                        "left_ankle": {"min_angle": 5.0, "max_angle": 25.0,
                                       "axis": "sagittal", "priority": "primary"},
                        "left_knee": {"min_angle": 150.0, "max_angle": 180.0,
                                      "axis": "sagittal", "priority": "secondary"},
                    },
                    red_flags=[
                        {"condition": "bilateral_asymmetry > 20",
                         "action": "warn", "reason": "Compensation pattern detected"},
                    ],
                    patient_instructions=(
                        "Stand on your affected leg with a slight knee bend. "
                        "Hold for 30 seconds. Focus on keeping the ankle stable."
                    ),
                    difficulty="intermediate",
                    safety_warnings=["Stand near a wall for safety"],
                ),
            ],
        ),
    ],
)


# ── Plan adaptation (JSON Patch) ──────────────────────────────────────────────

# Typical adaptation: reduce reps due to low quality scores
VALID_PATCH_RESPONSE: list[dict] = [
    {"op": "replace", "path": "/phases/0/exercises/0/reps", "value": 8},
    {"op": "replace", "path": "/phases/0/exercises/0/rest_seconds", "value": 45},
]

# No-change response (stable metrics)
EMPTY_PATCH_RESPONSE: list[dict] = []

# Progression response: increase difficulty
PROGRESSION_PATCH_RESPONSE: list[dict] = [
    {"op": "replace", "path": "/phases/0/exercises/0/reps", "value": 12},
    {"op": "replace", "path": "/phases/0/exercises/1/sets", "value": 3},
    {"op": "replace", "path": "/current_phase", "value": 2},
]


# ── Red-flag escalation ───────────────────────────────────────────────────────

VALID_RED_FLAG_RESPONSE: dict = {
    "severity":               "stop",
    "immediate_action":       (
        "Please stop the exercise and rest. "
        "Your physiotherapist has been notified."
    ),
    "clinician_note":         (
        "Patient reported sudden pain spike to 8/10 during seated ankle circles. "
        "Previous average was 3.2/10. Possible acute re-injury. "
        "Recommend same-day assessment."
    ),
    "session_recommendation": "rest_and_reassess",
}

WARN_RED_FLAG_RESPONSE: dict = {
    "severity":               "warn",
    "immediate_action":       "Slow down and focus on keeping your ankle aligned.",
    "clinician_note":         "Minor compensation pattern detected. Monitor next session.",
    "session_recommendation": "continue_with_caution",
}


# ── Feedback messages ─────────────────────────────────────────────────────────

VALID_FEEDBACK_RESPONSE: str = (
    "Push your ankle a little further to reach the full range of motion."
)

FEEDBACK_RESPONSES: dict[str, str] = {
    "ankle_insufficient_range":  "Push your ankle a little further to reach the full range.",
    "knee_valgus":               "Push your knee outward to align with your toes.",
    "lumbar_hyperextension":     "Gently flatten your lower back and engage your core.",
    "shoulder_elevation":        "Relax your shoulders down away from your ears.",
    "bilateral_asymmetry":       "Distribute your weight evenly across both sides.",
}