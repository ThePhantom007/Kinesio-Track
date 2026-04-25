"""
System prompt and builder for red-flag escalation responses.

Claude receives:
  - The trigger type and context (what was detected)
  - The patient's current exercise and pain state
  - The plan's escalation_criteria

Claude must return a structured JSON object with:
  - severity:              warn | stop | seek_care
  - immediate_action:      patient-facing plain-language instruction
  - clinician_note:        clinical context for the assigned clinician
  - session_recommendation: what to do with the current session

Latency expectation: this is time-sensitive — the patient is mid-session.
max_tokens should be capped at ~300 for this call type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai.prompt_templates.base import dedent, inject_patient_context


# ── System prompt ─────────────────────────────────────────────────────────────

RED_FLAG_SYSTEM_PROMPT = dedent("""
    You are a clinical safety AI in the Kinesio-Track physiotherapy platform.
    A potentially dangerous condition has been detected during a patient's
    exercise session.  Your response will be delivered in real time.

    OUTPUT FORMAT
    ─────────────
    Return ONLY valid JSON, no markdown, no commentary.

    {
      "severity": "warn" | "stop" | "seek_care",
      "immediate_action": string,        // patient-facing, plain English, ≤ 40 words
      "clinician_note": string,          // clinical context for the assigned clinician
      "session_recommendation": string   // one of: "continue_with_caution" | "rest_and_reassess" | "stop_session" | "seek_emergency_care"
    }

    SEVERITY DEFINITIONS
    ────────────────────
    warn:        Minor concern — patient can continue with caution and a correction cue.
                 Use for: form deviation near threshold, mild pain increase (1–2 points).

    stop:        Patient must stop the current exercise and rest.
                 Use for: significant pain spike (3+ points), moderate compensation pattern,
                 sharp or unusual pain description, ROM well outside safe range.

    seek_care:   Patient must stop all activity immediately and contact a clinician or
                 seek emergency care.
                 Use for: severe acute pain (8+/10), suspected injury exacerbation,
                 neurological symptoms (numbness, tingling), joint locking.

    TONE RULES
    ──────────
    - immediate_action must be calm, clear, and non-alarming unless severity=seek_care.
    - Do NOT use medical jargon in immediate_action.
    - clinician_note may use clinical terminology.
    - Always acknowledge what the patient did right before the corrective instruction
      in warn-level messages.
""")


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class RedFlagContext:
    """Assembled by red_flag_monitor service before calling Claude."""

    trigger_type: str                            # e.g. "pain_spike", "bilateral_asymmetry"
    trigger_context: dict[str, Any]              # structured trigger data

    # Current exercise state
    exercise_name: str
    exercise_slug: str
    current_pain_score: int | None
    previous_avg_pain: float | None

    # Patient context
    age: int | None
    activity_level: str | None
    body_part: str                               # the injury body part

    # Session state
    session_reps_completed: int = 0
    session_quality_score: float | None = None

    # Plan escalation criteria (from exercise_plan.escalation_criteria)
    escalation_criteria: list[dict[str, Any]] = field(default_factory=list)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_red_flag_prompt(ctx: RedFlagContext) -> str:
    """Build the user-turn prompt for red-flag escalation."""

    trigger_lines = [f"  {k}: {v}" for k, v in ctx.trigger_context.items()]
    trigger_detail = "\n".join(trigger_lines) if trigger_lines else "  (no additional detail)"

    pain_change = ""
    if ctx.current_pain_score is not None and ctx.previous_avg_pain is not None:
        delta = ctx.current_pain_score - ctx.previous_avg_pain
        sign  = "+" if delta > 0 else ""
        pain_change = f"  Pain change from baseline: {sign}{delta:.1f} points"

    escalation_block = ""
    if ctx.escalation_criteria:
        lines = ["Plan escalation criteria (for reference):"]
        for c in ctx.escalation_criteria:
            lines.append(f"  [{c.get('action', '?')}] {c.get('trigger', '?')} — {c.get('reason', '')}")
        escalation_block = "\n".join(lines)

    return dedent(f"""
        RED FLAG TRIGGER DETECTED
        ─────────────────────────
        Trigger type:     {ctx.trigger_type}
        Exercise:         {ctx.exercise_name} ({ctx.exercise_slug})
        Body part:        {ctx.body_part}
        Patient age:      {ctx.age or 'unknown'}
        Activity level:   {ctx.activity_level or 'unknown'}

        TRIGGER DETAIL
        ──────────────
        {trigger_detail}

        SESSION STATE
        ─────────────
        Reps completed this session: {ctx.session_reps_completed}
        Session quality score:       {ctx.session_quality_score or 'N/A'}
        Current pain score:          {ctx.current_pain_score or 'not reported'}/10
        Previous avg pain score:     {ctx.previous_avg_pain or 'N/A'}
        {pain_change}

        {escalation_block}

        Return the JSON escalation response as specified.
    """)