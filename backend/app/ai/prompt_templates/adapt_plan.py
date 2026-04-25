"""
System prompt and builder for post-session plan adaptation.

Claude receives:
  - The current plan structure (exercises + phases)
  - Session metrics from the last N sessions
  - Adaptation thresholds from config

Claude must return either:
  - An RFC 6902 JSON Patch array describing the minimal set of changes, or
  - An empty array [] if no adaptation is needed.

Using JSON Patch (not a full plan replacement) means:
  - The DB write is atomic and auditable (each patch op is logged)
  - Unchanged exercises are never touched
  - Clinician overrides on specific exercises are preserved unless explicitly patched
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.prompt_templates.base import (
    dedent,
    format_exercise_list,
    format_session_metrics,
    inject_patient_context,
    today_str,
)


# ── System prompt ─────────────────────────────────────────────────────────────

ADAPT_PLAN_SYSTEM_PROMPT = dedent("""
    You are an expert physiotherapist AI assistant in the Kinesio-Track platform.
    Your role is to adapt an existing exercise plan based on recent session data.

    OUTPUT FORMAT
    ─────────────
    Return ONLY a valid RFC 6902 JSON Patch array, or an empty array [] if no
    changes are needed.  No markdown, no commentary, no code fences.

    Example of a valid response (two changes):
    [
      {"op": "replace", "path": "/phases/0/exercises/1/reps", "value": 8},
      {"op": "replace", "path": "/phases/0/exercises/1/sets", "value": 2}
    ]

    Example of no changes needed:
    []

    ADAPTATION RULES
    ────────────────
    PROGRESSION (apply when avg_quality_score >= 78 over last 3 sessions AND pain <= 4):
      - Increase reps by 2–3 OR sets by 1
      - Advance to next phase if progression_criteria met
      - Introduce intermediate-difficulty variants

    REGRESSION (apply when avg_quality_score < 45 over last 3 sessions OR pain >= 7):
      - Reduce reps by 2 OR sets by 1 (never below sets=1, reps=3)
      - Revert to beginner difficulty
      - Add additional rest_seconds (+15–30)

    MAINTENANCE (apply when metrics are stable within acceptable range):
      - Return []
      - Do NOT make changes for the sake of making changes

    CONSTRAINTS
    ───────────
    - Only use "replace" and "add" operations; never use "remove" on entire exercises
    - Paths must reference valid locations in the plan JSON structure provided
    - Never change landmark_rules unless explicitly reducing range due to pain regression
    - Never change slug values
    - If phase advancement is warranted, set /current_phase to the next phase number
    - Maximum 8 patch operations per adaptation — keep changes surgical
""")


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class AdaptationContext:
    """Assembled by plan_adapter service before calling Claude."""

    current_plan: dict[str, Any]          # serialised ExercisePlan
    current_exercises: list[dict[str, Any]]  # flattened exercise list
    session_metrics: list[dict[str, Any]]    # last N session metric dicts
    avg_quality_score: float
    avg_pain_score: float
    completion_rate: float                 # 0.0–1.0
    sessions_analysed: int
    age: int | None
    activity_level: str | None
    mobility_notes: str | None


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_adapt_prompt(ctx: AdaptationContext) -> str:
    """
    Build the user-turn prompt for plan adaptation.

    Only the delta context is included — not the full plan JSON — to keep
    the prompt compact and focused on what Claude needs to decide.
    """
    patient_ctx = inject_patient_context(
        age=ctx.age,
        activity_level=ctx.activity_level,
        mobility_notes=ctx.mobility_notes,
        contraindications=[],
    )

    exercise_list = format_exercise_list(ctx.current_exercises)
    metrics_block  = format_session_metrics(ctx.session_metrics)

    return dedent(f"""
        {patient_ctx}

        CURRENT PLAN SUMMARY
        ────────────────────
        Current phase:    {ctx.current_plan.get('current_phase', '?')}
        Total phases:     {len(ctx.current_plan.get('phases', []))}
        Recovery target:  {ctx.current_plan.get('recovery_target_days', '?')} days

        EXERCISES IN CURRENT PHASE
        ──────────────────────────
        {exercise_list}

        {metrics_block}

        AGGREGATE METRICS ({ctx.sessions_analysed} session(s)):
        ───────────────────────────────────────
        Average quality score:  {ctx.avg_quality_score:.1f}/100
        Average pain score:     {ctx.avg_pain_score:.1f}/10
        Average completion:     {ctx.completion_rate*100:.0f}%

        Today's date: {today_str()}

        Based on the session data above, return an RFC 6902 JSON Patch array
        describing the minimal changes to improve this plan, or [] if no
        changes are needed.  Paths must reference the plan structure shown.
    """)