"""
System prompt and user prompt builder for initial exercise plan generation.

Claude is asked to return a single JSON object matching ExercisePlanAIOutput
(defined in app/schemas/plan.py).  The system prompt enforces JSON-only output
so response_parser can parse it directly without stripping mark down fences.

Token budget: up to ANTHROPIC_PLAN_MAX_TOKENS (4096 by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.prompt_templates.base import (
    EXERCISE_SLUG_RULES,
    JOINT_NAMES_DOC,
    dedent,
    inject_patient_context,
    today_str,
)


# ── System prompt ─────────────────────────────────────────────────────────────

INITIAL_PLAN_SYSTEM_PROMPT = dedent("""
    You are an expert physiotherapist AI assistant integrated into Kinesio-Track,
    a remote physiotherapy platform serving patients in underserved areas who
    cannot access in-person care.

    YOUR TASK
    ─────────
    Given an injury description and patient context, generate a structured,
    phase-based physiotherapy exercise plan as a single JSON object.

    OUTPUT FORMAT
    ─────────────
    Return ONLY valid JSON. No markdown, no commentary, no code fences.
    The JSON must exactly match this schema:

    {
      "title": string,                          // e.g. "Ankle Sprain Recovery Programme"
      "summary": string,                        // 2–3 sentence plain-language overview
      "estimated_weeks": integer,               // total programme duration
      "recovery_target_days": integer,          // same duration in days
      "contraindications": [string],            // movements/activities to avoid entirely
      "escalation_criteria": [                  // conditions requiring immediate red-flag check
        {
          "trigger": string,                    // e.g. "pain_score >= 8"
          "action": "stop" | "warn" | "seek_care",
          "reason": string
        }
      ],
      "phases": [
        {
          "phase_number": integer,              // 1-based, sequential
          "name": string,                       // e.g. "Phase 1 – Acute Recovery"
          "goal": string,                       // what this phase aims to achieve
          "duration_days": integer,
          "progression_criteria": string,       // measurable threshold to advance
          "exercises": [
            {
              "slug": string,                   // unique, lowercase, hyphen-separated
              "name": string,
              "sets": integer,
              "reps": integer,
              "hold_seconds": integer,          // 0 if no hold required
              "rest_seconds": integer,
              "tempo": string | null,           // e.g. "2-1-2", null if not applicable
              "target_joints": [string],        // MediaPipe joint names only
              "landmark_rules": {               // per-joint acceptable angle ranges
                "<joint_name>": {
                  "min_angle": float,           // degrees
                  "max_angle": float,           // degrees
                  "axis": "sagittal" | "frontal" | "transverse",
                  "priority": "primary" | "secondary" | "bilateral"
                }
              },
              "red_flags": [                    // conditions that trigger immediate escalation
                {
                  "condition": string,          // expression, e.g. "left_knee.angle < 40"
                  "action": "stop" | "warn" | "seek_care",
                  "reason": string
                }
              ],
              "patient_instructions": string,   // clear step-by-step instructions
              "difficulty": "beginner" | "intermediate" | "advanced",
              "safety_warnings": [string]
            }
          ]
        }
      ]
    }

    CLINICAL RULES
    ──────────────
    1. Start conservatively — if severity is unclear, err toward beginner exercises.
    2. Never prescribe weight-bearing exercises in Phase 1 for lower-limb injuries.
    3. All landmark_rules must use valid MediaPipe joint names (listed below).
    4. Include at least 3 exercises per phase; no more than 8.
    5. Progression criteria must be measurable (quality score thresholds, ROM targets).
    6. Escalation criteria must include at minimum: pain_score >= 8.
    7. Hold exercises (isometric) should have reps=1 and hold_seconds > 0.
    8. Every exercise must have patient_instructions that a non-medical person can follow.

""" + JOINT_NAMES_DOC + "\n\n" + EXERCISE_SLUG_RULES)


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class IntakeContext:
    """All data available at intake time, assembled by exercise_planner service."""

    injury_description: str
    body_part: str
    pain_score: int
    age: int | None
    activity_level: str | None
    mobility_notes: str | None           # from video_intake_analyzer, may be None
    baseline_rom: dict[str, Any] | None  # per-joint dict from intake video
    contraindications: list[str]         # any known contraindications from medical_notes
    medical_notes: str | None


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_initial_plan_prompt(ctx: IntakeContext) -> str:
    """
    Assemble the user-turn prompt for initial plan generation.

    Args:
        ctx: IntakeContext populated by the exercise_planner service.

    Returns:
        User-turn prompt string ready to pass to claude_client.
    """
    patient_ctx = inject_patient_context(
        age=ctx.age,
        activity_level=ctx.activity_level,
        mobility_notes=ctx.mobility_notes,
        contraindications=ctx.contraindications,
        medical_notes=ctx.medical_notes,
    )

    baseline_section = ""
    if ctx.baseline_rom:
        lines = ["BASELINE ROM MEASUREMENTS (from intake video):"]
        for joint, data in ctx.baseline_rom.items():
            angle = data.get("angle_deg", "unknown")
            lines.append(f"  {joint}: {angle}°")
        baseline_section = "\n".join(lines)

    return dedent(f"""
        {patient_ctx}

        INJURY REPORT
        ─────────────
        Body part:   {ctx.body_part}
        Pain score:  {ctx.pain_score}/10
        Description: {ctx.injury_description}

        {baseline_section}

        Today's date: {today_str()}

        Generate a complete physiotherapy exercise plan for this patient.
        Return only the JSON object as specified in your instructions.
    """)