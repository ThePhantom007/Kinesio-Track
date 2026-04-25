"""
Shared prompt-building utilities imported by all four template modules.

Design principles
-----------------
- Every patient-facing prompt is injected with a minimal patient context
  block so Claude can calibrate language complexity and exercise difficulty.
- History truncation ensures we never exceed the context window on long
  treatment courses.
- All builders return plain strings; the claude_client is responsible for
  wrapping them in the Anthropic messages list format.
- No business logic lives here — these are pure string builders.
"""

from __future__ import annotations

import textwrap
from datetime import date
from typing import Any


# ── Patient context injection ─────────────────────────────────────────────────

def inject_patient_context(
    *,
    age: int | None,
    activity_level: str | None,
    mobility_notes: str | None,
    contraindications: list[str] | None,
    medical_notes: str | None = None,
) -> str:
    """
    Produces a standardised patient context block prepended to plan-generation
    and adaptation prompts.  Omits fields that are None to keep prompts lean.

    Example output:
        PATIENT CONTEXT
        ───────────────
        Age: 34
        Activity level: moderately_active
        Baseline mobility notes: Limited dorsiflexion in left ankle (~18°).
        Contraindications: Avoid full weight-bearing on left foot.
    """
    lines = ["PATIENT CONTEXT", "───────────────"]
    if age is not None:
        lines.append(f"Age: {age}")
    if activity_level:
        lines.append(f"Activity level: {activity_level}")
    if mobility_notes:
        lines.append(f"Baseline mobility notes: {mobility_notes}")
    if contraindications:
        ci = "; ".join(contraindications)
        lines.append(f"Contraindications: {ci}")
    if medical_notes:
        lines.append(f"Clinician notes: {medical_notes}")
    return "\n".join(lines)


# ── Exercise list formatter ────────────────────────────────────────────────────

def format_exercise_list(exercises: list[dict[str, Any]]) -> str:
    """
    Formats a list of exercise dicts for inclusion in adaptation prompts so
    Claude can reference existing exercises by slug when producing a JSON Patch.

    Each exercise dict is expected to have: slug, name, sets, reps, phase_number.
    """
    if not exercises:
        return "No exercises currently in plan."
    lines = []
    for ex in exercises:
        lines.append(
            f"  - [{ex.get('slug', 'unknown')}] {ex.get('name', 'Unknown')} "
            f"(Phase {ex.get('phase_number', '?')}, "
            f"{ex.get('sets', '?')}×{ex.get('reps', '?')})"
        )
    return "\n".join(lines)


# ── Session metrics formatter ──────────────────────────────────────────────────

def format_session_metrics(metrics: list[dict[str, Any]], n: int = 5) -> str:
    """
    Formats the last N session metric dicts for inclusion in adaptation prompts.

    Each metric dict is expected to have:
        session_date, avg_quality_score, completion_pct,
        post_session_pain, peak_rom_degrees.
    """
    recent = metrics[-n:] if len(metrics) > n else metrics
    if not recent:
        return "No session history available."
    lines = [f"LAST {len(recent)} SESSION(S):", "───────────────────────────"]
    for m in recent:
        date_str = m.get("session_date", "unknown date")
        quality  = m.get("avg_quality_score")
        pain     = m.get("post_session_pain")
        comp     = m.get("completion_pct")
        rom      = m.get("peak_rom_degrees")

        parts = [f"  {date_str}:"]
        if quality is not None:
            parts.append(f"quality={quality:.1f}/100")
        if pain is not None:
            parts.append(f"pain={pain}/10")
        if comp is not None:
            parts.append(f"completion={comp*100:.0f}%")
        if rom is not None:
            parts.append(f"peak_ROM={rom:.1f}°")
        lines.append("  ".join(parts))
    return "\n".join(lines)


# ── History truncation ────────────────────────────────────────────────────────

def truncate_history(
    messages: list[dict[str, str]],
    max_tokens: int = 6000,
    chars_per_token: float = 4.0,
) -> list[dict[str, str]]:
    """
    Trims a conversation history list from the front (oldest first) until the
    estimated total token count is below *max_tokens*.

    Keeps system messages intact; only trims user/assistant turns.

    Args:
        messages:       List of {"role": ..., "content": ...} dicts.
        max_tokens:     Approximate token budget for the history.
        chars_per_token: Rough characters-per-token estimate (4 is conservative).

    Returns:
        Trimmed message list.  Always includes at least the most recent exchange.
    """
    budget_chars = int(max_tokens * chars_per_token)
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs   = [m for m in messages if m["role"] != "system"]

    total_chars = sum(len(m["content"]) for m in system_msgs)
    kept: list[dict[str, str]] = []

    for msg in reversed(conv_msgs):
        msg_chars = len(msg["content"])
        if total_chars + msg_chars > budget_chars and kept:
            break   # Always keep at least the last exchange
        kept.insert(0, msg)
        total_chars += msg_chars

    return system_msgs + kept


# ── Shared JSON schema constants ───────────────────────────────────────────────

JOINT_NAMES_DOC = """\
Valid MediaPipe joint names for landmark_rules keys:
  left_ankle, right_ankle, left_knee, right_knee,
  left_hip, right_hip, left_shoulder, right_shoulder,
  left_elbow, right_elbow, left_wrist, right_wrist,
  neck, lumbar_spine, thoracic_spine
"""

EXERCISE_SLUG_RULES = (
    "Exercise slugs must be lowercase, hyphen-separated, unique within the plan, "
    "e.g. 'seated-ankle-circles', 'standing-calf-raises'."
)


# ── Date helpers ──────────────────────────────────────────────────────────────

def today_str() -> str:
    return date.today().isoformat()


def dedent(text: str) -> str:
    """Strip common leading whitespace from a multiline string."""
    return textwrap.dedent(text).strip()