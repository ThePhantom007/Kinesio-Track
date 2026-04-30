"""
Unit tests for app/ai/response_parser.py.

Validates that the parser correctly accepts valid Claude output, rejects
malformed responses with useful error details, and produces corrective
retry prompts on failure.  No network calls.
"""

from __future__ import annotations

import json

import pytest

from app.ai.response_parser import (
    build_correction_prompt,
    validate_feedback_message,
    validate_initial_plan,
    validate_plan_patch,
    validate_red_flag_response,
)
from app.core.exceptions import PlanValidationError
from tests.fixtures.mock_claude_responses import VALID_PLAN_RESPONSE


# ── validate_initial_plan ─────────────────────────────────────────────────────

class TestValidateInitialPlan:

    def _valid_json(self) -> str:
        return VALID_PLAN_RESPONSE.model_dump_json()

    def test_valid_plan_returns_model(self):
        result = validate_initial_plan(self._valid_json())
        assert result.title == VALID_PLAN_RESPONSE.title
        assert len(result.phases) == len(VALID_PLAN_RESPONSE.phases)

    def test_strips_markdown_fences(self):
        raw = f"```json\n{self._valid_json()}\n```"
        result = validate_initial_plan(raw)
        assert result.title == VALID_PLAN_RESPONSE.title

    def test_strips_bare_code_fences(self):
        raw = f"```\n{self._valid_json()}\n```"
        result = validate_initial_plan(raw)
        assert result.title == VALID_PLAN_RESPONSE.title

    def test_invalid_json_raises_plan_validation_error(self):
        with pytest.raises(PlanValidationError) as exc_info:
            validate_initial_plan("this is not json {{{")
        assert "invalid JSON" in str(exc_info.value.message).lower() or \
               exc_info.value.error_code == "plan_validation_error"

    def test_missing_required_field_raises(self):
        data = json.loads(self._valid_json())
        del data["phases"]
        with pytest.raises(PlanValidationError) as exc_info:
            validate_initial_plan(json.dumps(data))
        assert exc_info.value.detail is not None

    def test_wrong_type_raises_with_diff(self):
        data = json.loads(self._valid_json())
        data["estimated_weeks"] = "six"   # should be int
        with pytest.raises(PlanValidationError) as exc_info:
            validate_initial_plan(json.dumps(data))
        assert exc_info.value.detail is not None

    def test_non_sequential_phases_raises(self):
        data = json.loads(self._valid_json())
        data["phases"][0]["phase_number"] = 5   # should be 1
        with pytest.raises(PlanValidationError):
            validate_initial_plan(json.dumps(data))

    def test_empty_phases_raises(self):
        data = json.loads(self._valid_json())
        data["phases"] = []
        with pytest.raises(PlanValidationError):
            validate_initial_plan(json.dumps(data))


# ── validate_plan_patch ───────────────────────────────────────────────────────

class TestValidatePlanPatch:

    def test_valid_patch_returns_list(self):
        raw = json.dumps([
            {"op": "replace", "path": "/phases/0/exercises/0/reps", "value": 8},
        ])
        result = validate_plan_patch(raw)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_empty_array_is_valid(self):
        result = validate_plan_patch("[]")
        assert result == []

    def test_not_an_array_raises(self):
        with pytest.raises(PlanValidationError):
            validate_plan_patch('{"op": "replace"}')

    def test_missing_op_raises(self):
        raw = json.dumps([{"path": "/phases/0/exercises/0/reps", "value": 8}])
        with pytest.raises(PlanValidationError) as exc_info:
            validate_plan_patch(raw)
        assert "missing" in str(exc_info.value.detail).lower()

    def test_missing_path_raises(self):
        raw = json.dumps([{"op": "replace", "value": 8}])
        with pytest.raises(PlanValidationError):
            validate_plan_patch(raw)

    def test_invalid_op_raises(self):
        raw = json.dumps([{"op": "destroy", "path": "/phases/0", "value": None}])
        with pytest.raises(PlanValidationError):
            validate_plan_patch(raw)

    def test_remove_op_is_rejected(self):
        raw = json.dumps([{"op": "remove", "path": "/phases/0/exercises/0"}])
        with pytest.raises(PlanValidationError):
            validate_plan_patch(raw)

    def test_multiple_valid_ops(self):
        raw = json.dumps([
            {"op": "replace", "path": "/phases/0/exercises/0/reps",         "value": 8},
            {"op": "replace", "path": "/phases/0/exercises/0/rest_seconds",  "value": 45},
            {"op": "replace", "path": "/current_phase",                       "value": 2},
        ])
        result = validate_plan_patch(raw)
        assert len(result) == 3


# ── validate_red_flag_response ────────────────────────────────────────────────

class TestValidateRedFlagResponse:

    def _valid(self) -> dict:
        return {
            "severity":               "stop",
            "immediate_action":       "Please stop and rest.",
            "clinician_note":         "Pain spike detected. Reassessment needed.",
            "session_recommendation": "rest_and_reassess",
        }

    def test_valid_response_returns_dict(self):
        result = validate_red_flag_response(json.dumps(self._valid()))
        assert result["severity"] == "stop"

    def test_missing_severity_raises(self):
        data = self._valid()
        del data["severity"]
        with pytest.raises(PlanValidationError):
            validate_red_flag_response(json.dumps(data))

    def test_invalid_severity_raises(self):
        data = self._valid()
        data["severity"] = "panic"
        with pytest.raises(PlanValidationError):
            validate_red_flag_response(json.dumps(data))

    def test_all_valid_severities(self):
        for severity in ("warn", "stop", "seek_care"):
            data = self._valid()
            data["severity"] = severity
            result = validate_red_flag_response(json.dumps(data))
            assert result["severity"] == severity

    def test_invalid_session_recommendation_raises(self):
        data = self._valid()
        data["session_recommendation"] = "just_walk_it_off"
        with pytest.raises(PlanValidationError):
            validate_red_flag_response(json.dumps(data))

    def test_all_valid_recommendations(self):
        for rec in ("continue_with_caution", "rest_and_reassess",
                    "stop_session", "seek_emergency_care"):
            data = self._valid()
            data["session_recommendation"] = rec
            result = validate_red_flag_response(json.dumps(data))
            assert result["session_recommendation"] == rec


# ── validate_feedback_message ─────────────────────────────────────────────────

class TestValidateFeedbackMessage:

    def test_short_clean_message_returned(self):
        msg = "Push your ankle a little further to reach the full range."
        result = validate_feedback_message(msg)
        assert result == msg

    def test_strips_surrounding_quotes(self):
        result = validate_feedback_message('"Push your knee out."')
        assert result == "Push your knee out."

    def test_strips_whitespace(self):
        result = validate_feedback_message("  Keep your back straight.  ")
        assert result == "Keep your back straight."

    def test_long_message_truncated_to_first_sentence(self):
        long_msg = (
            "Your form looks good overall but try to push your ankle "
            "further for a better range. Also remember to breathe steadily "
            "and keep your core engaged throughout the movement. "
            "You are doing great work today!"
        )
        result = validate_feedback_message(long_msg)
        word_count = len(result.split())
        assert word_count <= 30   # truncated or still within generous limit

    def test_empty_returns_safe_fallback(self):
        result = validate_feedback_message("")
        assert len(result) > 0

    def test_strips_code_fences(self):
        result = validate_feedback_message("```Push your knee out.```")
        assert "```" not in result


# ── build_correction_prompt ───────────────────────────────────────────────────

class TestBuildCorrectionPrompt:

    def test_includes_original_prompt(self):
        original = "Generate a plan for an ankle sprain."
        exc = PlanValidationError("bad", detail={"diff": "  Field 'phases': missing"})
        result = build_correction_prompt(original, exc)
        assert original in result

    def test_includes_diff_details(self):
        exc = PlanValidationError("bad", detail={"diff": "  Field 'title': required"})
        result = build_correction_prompt("original", exc)
        assert "title" in result

    def test_includes_correction_instruction(self):
        exc = PlanValidationError("bad", detail={"diff": "some diff"})
        result = build_correction_prompt("original", exc)
        assert "CORRECTION" in result.upper() or "corrected" in result.lower()

    def test_handles_no_detail(self):
        exc = PlanValidationError("schema error")
        result = build_correction_prompt("original", exc)
        assert "original" in result
        assert len(result) > 10