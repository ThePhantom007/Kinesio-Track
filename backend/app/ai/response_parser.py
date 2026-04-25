"""
Validates and parses every raw Claude JSON response before it reaches the DB
or is sent to a patient.

Responsibilities
----------------
  1. Strip any accidental mark down fences Claude might add despite instructions.
  2. Parse the JSON string into a Python dict.
  3. Validate against the appropriate Pydantic schema.
  4. On failure: raise PlanValidationError with a human-readable diff so that
     claude_client can construct a targeted corrective retry prompt.
  5. Return the validated Pydantic model instance — never raw dicts.

Each public ``validate_*`` method corresponds to one Claude call type.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.core.exceptions import PlanValidationError
from app.core.logging import get_logger
from app.schemas.plan import ExercisePlanAIOutput

log = get_logger(__name__)

# ── Regex to strip mark down code fences ──────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ── JSON Patch op schema (lightweight, no external dep) ───────────────────────
_VALID_OPS = {"add", "replace", "move", "copy", "test"}
_REQUIRED_PATCH_KEYS = {"op", "path"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


def _parse_json(raw: str, call_type: str) -> Any:
    """
    Parse *raw* as JSON.  Raises PlanValidationError (not json.JSONDecodeError)
    so the caller always catches the same exception type.
    """
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("json_parse_failed", call_type=call_type, error=str(exc), raw_preview=raw[:200])
        raise PlanValidationError(
            f"Claude returned invalid JSON for '{call_type}'.",
            detail={"json_error": str(exc), "raw_preview": raw[:300]},
        ) from exc


def _build_validation_diff(error: ValidationError) -> str:
    """
    Convert a Pydantic ValidationError into a concise diff string suitable for
    inclusion in a corrective retry prompt.
    """
    lines = []
    for err in error.errors():
        loc  = " → ".join(str(p) for p in err["loc"])
        msg  = err["msg"]
        val  = err.get("input", "<missing>")
        lines.append(f"  Field '{loc}': {msg} (got: {val!r})")
    return "\n".join(lines)


# ── Public validators ─────────────────────────────────────────────────────────

def validate_initial_plan(raw: str) -> ExercisePlanAIOutput:
    """
    Parse and validate a raw Claude response for initial plan generation.

    Returns:
        Validated ExercisePlanAIOutput instance.

    Raises:
        PlanValidationError: JSON parse error or schema violation.
            ``detail["diff"]`` contains the field-level error summary for
            constructing a corrective retry prompt in claude_client.
    """
    data = _parse_json(raw, "initial_plan")

    try:
        plan = ExercisePlanAIOutput.model_validate(data)
    except ValidationError as exc:
        diff = _build_validation_diff(exc)
        log.warning(
            "plan_validation_failed",
            error_count=exc.error_count(),
            diff=diff,
        )
        raise PlanValidationError(
            f"Claude's plan failed schema validation ({exc.error_count()} error(s)).",
            detail={"diff": diff, "raw_preview": raw[:500]},
        ) from exc

    log.info(
        "plan_validated",
        phases=len(plan.phases),
        total_exercises=sum(len(p.exercises) for p in plan.phases),
        estimated_weeks=plan.estimated_weeks,
    )
    return plan


def validate_plan_patch(raw: str) -> list[dict[str, Any]]:
    """
    Parse and validate a raw Claude response for plan adaptation (JSON Patch).

    Accepts either a JSON Patch array or an empty array [].

    Returns:
        List of RFC 6902 patch operation dicts (maybe empty).

    Raises:
        PlanValidationError: Invalid JSON, not an array, or malformed patch ops.
    """
    data = _parse_json(raw, "adapt_plan")

    if not isinstance(data, list):
        raise PlanValidationError(
            "Plan adaptation response must be a JSON array.",
            detail={"got_type": type(data).__name__, "raw_preview": raw[:300]},
        )

    errors: list[str] = []
    for i, op in enumerate(data):
        if not isinstance(op, dict):
            errors.append(f"  Op[{i}]: must be a JSON object, got {type(op).__name__}")
            continue
        missing = _REQUIRED_PATCH_KEYS - op.keys()
        if missing:
            errors.append(f"  Op[{i}]: missing required keys {missing}")
        if "op" in op and op["op"] not in _VALID_OPS:
            errors.append(f"  Op[{i}]: invalid op '{op['op']}', must be one of {_VALID_OPS}")
        if op.get("op") == "remove":
            errors.append(f"  Op[{i}]: 'remove' is not permitted in plan adaptations")

    if errors:
        diff = "\n".join(errors)
        raise PlanValidationError(
            f"Plan patch contains {len(errors)} invalid operation(s).",
            detail={"diff": diff},
        )

    log.info("plan_patch_validated", op_count=len(data))
    return data


def validate_red_flag_response(raw: str) -> dict[str, Any]:
    """
    Parse and validate a raw Claude response for red-flag escalation.

    Returns:
        Dict with keys: severity, immediate_action, clinician_note,
        session_recommendation.

    Raises:
        PlanValidationError: Missing required fields or invalid severity value.
    """
    data = _parse_json(raw, "red_flag")

    required = {"severity", "immediate_action", "clinician_note", "session_recommendation"}
    missing = required - data.keys()
    if missing:
        raise PlanValidationError(
            f"Red flag response missing required fields: {missing}.",
            detail={"missing": list(missing)},
        )

    valid_severities = {"warn", "stop", "seek_care"}
    if data["severity"] not in valid_severities:
        raise PlanValidationError(
            f"Invalid severity '{data['severity']}'. Must be one of {valid_severities}.",
            detail={"got": data["severity"]},
        )

    valid_recommendations = {
        "continue_with_caution", "rest_and_reassess",
        "stop_session", "seek_emergency_care",
    }
    if data["session_recommendation"] not in valid_recommendations:
        raise PlanValidationError(
            f"Invalid session_recommendation '{data['session_recommendation']}'.",
            detail={"got": data["session_recommendation"]},
        )

    log.info("red_flag_response_validated", severity=data["severity"])
    return data


def validate_feedback_message(raw: str) -> str:
    """
    Validate a raw Claude response for real-time feedback.

    Claude is asked for a single plain-text sentence.  This validator:
      - Strips leading/trailing whitespace and quotes.
      - Enforces a 25-word maximum (slightly above the 20-word prompt limit
        to absorb minor over-runs without hard-failing).
      - Falls back to a generic safe message if validation fails, rather than
        raising — feedback failures must not break the live WebSocket loop.

    Returns:
        Validated correction sentence string.
    """
    text = raw.strip().strip('"').strip("'")

    # Strip any accidental fences
    text = _strip_fences(text)

    word_count = len(text.split())
    if word_count > 25:
        log.warning("feedback_too_long", word_count=word_count, text=text)
        # Truncate to first sentence if possible
        sentences = text.split(".")
        text = sentences[0].strip() + "."

    if not text:
        log.warning("feedback_empty_response")
        return "Focus on your form and move slowly and carefully."

    return text


# ── Corrective retry prompt builder ──────────────────────────────────────────

def build_correction_prompt(original_prompt: str, validation_error: PlanValidationError) -> str:
    """
    Construct a follow-up prompt for claude_client to retry after a
    PlanValidationError.  Includes the original prompt and the specific
    field-level diff so Claude can correct only the failing parts.

    Used by claude_client in its retry loop.
    """
    diff = validation_error.detail.get("diff", str(validation_error)) if validation_error.detail else str(validation_error)
    return (
        f"{original_prompt}\n\n"
        "CORRECTION REQUIRED\n"
        "───────────────────\n"
        "Your previous response failed schema validation.  "
        "Fix the following issues and return the corrected JSON:\n\n"
        f"{diff}\n\n"
        "Return ONLY the corrected JSON object with no other text."
    )