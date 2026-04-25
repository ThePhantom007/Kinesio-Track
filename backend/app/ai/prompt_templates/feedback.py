"""
System prompt and builder for real-time exercise correction messages.

This is the highest-frequency Claude call — fired on cache miss when the
pose_analyzer detects a joint angle violation.  Optimised for:
  - Minimum token usage  (max_tokens = 80)
  - Maximum clarity      (single sentence, ≤ 20 words)
  - Encouraging tone     (avoid alarm; build confidence)

Cache strategy
--------------
The feedback_generator service caches responses in Redis keyed by
(exercise_slug, error_type) with a 24-hour TTL.  On a cache hit,
Claude is never called.  This prompt only executes on a cache miss,
which typically means a new or rare error type for that exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.prompt_templates.base import dedent


# ── System prompt ─────────────────────────────────────────────────────────────

FEEDBACK_SYSTEM_PROMPT = dedent("""
    You are a real-time physiotherapy coach inside the Kinesio-Track app.
    A patient is exercising right now and their form needs a correction.

    OUTPUT FORMAT
    ─────────────
    Return ONLY a single correction sentence.
    - Maximum 20 words.
    - No markdown, no quotes, no punctuation other than a full stop.
    - Plain text only.

    TONE RULES
    ──────────
    - Warm, encouraging, coach-like — never clinical or alarming.
    - Use "you" not "the patient".
    - Use action verbs: "Straighten", "Relax", "Keep", "Bring", "Lower", "Lift".
    - Describe the correction, not the mistake.
      WRONG: "Your knee is collapsing inward."
      RIGHT: "Push your knee out to stay in line with your toes."

    DIFFICULTY CALIBRATION
    ──────────────────────
    beginner:     Simple, short words. Avoid anatomy terms.
    intermediate: May name the muscle group. Slightly more detail.
    advanced:     Can use precise anatomical cues and tempo references.
""")


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class FeedbackContext:
    """Assembled by feedback_generator service on Redis cache miss."""

    exercise_name: str
    exercise_slug: str
    error_type: str          # e.g. "knee_valgus", "lumbar_hyperextension"
    affected_joint: str      # MediaPipe joint name
    deviation_degrees: float
    deviation_direction: str  # "flexed" | "extended" | "abducted" | "adducted"
    difficulty: str           # "beginner" | "intermediate" | "advanced"
    patient_age: int | None = None


# ── Error type → plain description map ────────────────────────────────────────

_ERROR_DESCRIPTIONS: dict[str, str] = {
    "knee_valgus":               "knee collapsing inward (valgus)",
    "knee_varus":                "knee bowing outward (varus)",
    "lumbar_hyperextension":     "lower back arching excessively",
    "lumbar_flexion":            "lower back rounding",
    "shoulder_elevation":        "shoulder shrugging upward",
    "shoulder_protraction":      "shoulder rolling forward",
    "ankle_inversion":           "ankle rolling inward",
    "ankle_eversion":            "ankle rolling outward",
    "hip_drop":                  "hip dropping to one side",
    "neck_forward_flexion":      "chin jutting forward",
    "elbow_hyperextension":      "elbow locking out",
    "bilateral_asymmetry":       "uneven weight distribution between sides",
    "insufficient_range":        "not reaching the full prescribed range of motion",
    "excessive_range":           "moving beyond the safe range of motion",
}


def _describe_error(error_type: str) -> str:
    return _ERROR_DESCRIPTIONS.get(error_type, error_type.replace("_", " "))


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_feedback_prompt(ctx: FeedbackContext) -> str:
    """
    Build the user-turn prompt for a single correction message.

    Kept deliberately short — the system prompt already establishes all
    constraints; this turn just supplies the specific frame of reference.
    """
    error_desc = _describe_error(ctx.error_type)

    return dedent(f"""
        Exercise:   {ctx.exercise_name}
        Joint:      {ctx.affected_joint}
        Issue:      {error_desc}
        Direction:  {ctx.deviation_direction} by {ctx.deviation_degrees:.1f}°
        Difficulty: {ctx.difficulty}

        Write a single correction sentence for this patient.
    """)


# ── Cache key helper ──────────────────────────────────────────────────────────

def feedback_cache_key(exercise_slug: str, error_type: str, difficulty: str) -> str:
    """
    Redis key for caching a feedback message.
    Includes difficulty so beginners and advanced patients get appropriately
    calibrated messages from the same exercise/error combination.
    """
    return f"feedback_cache:{exercise_slug}:{error_type}:{difficulty}"