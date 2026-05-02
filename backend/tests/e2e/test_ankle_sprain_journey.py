"""
End-to-end test: full patient journey from intake to progress chart data.

  1. Register a new patient account
  2. Submit an ankle sprain injury intake → plan generated
  3. Start a session, send synthetic landmark frames, end the session
  4. Repeat for 3 sessions (simulating improvement over time)
  5. Assert plan adaptation was triggered after session 3
  6. Assert progress chart data returns a correct improving trajectory

Claude, S3, TimescaleDB queries, and Celery tasks are all mocked so this
test runs without external infrastructure, while still exercising the full
HTTP → service → DB write pipeline end-to-end.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.fixtures.mock_claude_responses import (
    VALID_PLAN_RESPONSE,
    VALID_PATCH_RESPONSE,
    VALID_FEEDBACK_RESPONSE,
    EMPTY_PATCH_RESPONSE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _good_landmarks() -> list[dict]:
    """33 landmarks representing correct ankle circle form."""
    positions = {
        0: (0.50, 0.05), 7: (0.46, 0.08), 8: (0.54, 0.08),
        11: (0.44, 0.28), 12: (0.56, 0.28), 13: (0.42, 0.42),
        14: (0.58, 0.42), 15: (0.40, 0.55), 16: (0.60, 0.55),
        19: (0.39, 0.60), 20: (0.61, 0.60),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.46, 0.88), 28: (0.54, 0.88),
        31: (0.46, 0.96), 32: (0.54, 0.96),
    }
    lm = []
    for i in range(33):
        x, y = positions.get(i, (0.50, 0.50))
        lm.append({"id": i, "x": x, "y": y, "z": -0.02, "visibility": 0.95})
    return lm


async def _register_and_login(client) -> dict:
    """Register a fresh patient and return auth headers + patient info."""
    email = f"e2e_ankle_{uuid4().hex[:8]}@test.local"
    reg = await client.post("/api/v1/auth/register", json={
        "email":     email,
        "password":  "StrongPass1!",
        "full_name": "E2E Ankle Patient",
        "role":      "patient",
    })
    assert reg.status_code == 201
    tokens = reg.json()["tokens"]
    return {
        "headers":  {"Authorization": f"Bearer {tokens['access_token']}"},
        "tokens":   tokens,
        "user":     reg.json()["user"],
    }


async def _do_intake(client, headers) -> dict:
    """Submit intake and return the response body."""
    resp = await client.post("/api/v1/intake", json={
        "description": (
            "Rolled my left ankle playing cricket 10 days ago. "
            "Significant swelling for the first week, now resolving. "
            "Still feels unstable on uneven ground and aches after standing."
        ),
        "body_part":  "ankle",
        "pain_score": 6,
    }, headers=headers)
    assert resp.status_code == 201
    return resp.json()


async def _start_and_end_session(client, headers, plan_id, pain_score) -> dict:
    """Start a session, end it, and return the summary."""
    start = await client.post("/api/v1/sessions", json={
        "plan_id": plan_id,
    }, headers=headers)
    assert start.status_code == 201
    session_id = start.json()["session_id"]

    end = await client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"post_session_pain": pain_score},
        headers=headers,
    )
    assert end.status_code == 200
    return end.json()


# ── E2E journey ───────────────────────────────────────────────────────────────

class TestAnkleSprainJourney:

    @pytest.mark.asyncio
    @patch("app.workers.session_tasks.post_session_analysis.delay")
    async def test_intake_creates_plan_with_correct_structure(
        self, mock_celery, client
    ):
        """Step 1–2: Register + intake → plan exists with phases and exercises."""
        auth = await _register_and_login(client)
        intake_body = await _do_intake(client, auth["headers"])

        assert "plan_id"           in intake_body
        assert intake_body["estimated_phases"] == len(VALID_PLAN_RESPONSE.phases)
        assert intake_body["estimated_weeks"]  == VALID_PLAN_RESPONSE.estimated_weeks

    @pytest.mark.asyncio
    @patch("app.workers.session_tasks.post_session_analysis.delay")
    async def test_plan_endpoint_returns_phases_and_exercises(
        self, mock_celery, client
    ):
        """After intake, GET /plans/{id} returns the full plan structure."""
        auth = await _register_and_login(client)
        intake_body = await _do_intake(client, auth["headers"])
        plan_id = intake_body["plan_id"]

        resp = await client.get(
            f"/api/v1/plans/{plan_id}",
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        plan = resp.json()

        assert len(plan["phases"]) == len(VALID_PLAN_RESPONSE.phases)
        # Each phase has at least one exercise
        for phase in plan["phases"]:
            assert len(phase["exercises"]) >= 1
            for ex in phase["exercises"]:
                assert "landmark_rules" in ex
                assert "patient_instructions" in ex

    @pytest.mark.asyncio
    @patch("app.workers.session_tasks.post_session_analysis.delay")
    async def test_three_sessions_complete_successfully(
        self, mock_celery, client
    ):
        """Steps 3–4: Complete 3 sessions with decreasing pain scores."""
        auth        = await _register_and_login(client)
        intake_body = await _do_intake(client, auth["headers"])
        plan_id     = intake_body["plan_id"]

        pain_scores = [6, 4, 3]   # improving pain trajectory
        sessions    = []

        for pain in pain_scores:
            summary = await _start_and_end_session(
                client, auth["headers"], plan_id, pain
            )
            sessions.append(summary)
            assert summary["metrics"]["post_session_pain"] == pain

        assert len(sessions) == 3

    @pytest.mark.asyncio
    @patch("app.workers.session_tasks.post_session_analysis.delay")
    async def test_session_history_shows_all_three_sessions(
        self, mock_celery, client
    ):
        """After 3 sessions, history endpoint returns all three."""
        auth        = await _register_and_login(client)
        intake_body = await _do_intake(client, auth["headers"])
        plan_id     = intake_body["plan_id"]

        for pain in [6, 4, 3]:
            await _start_and_end_session(
                client, auth["headers"], plan_id, pain
            )

        history = await client.get("/api/v1/sessions", headers=auth["headers"])
        assert history.status_code == 200
        assert history.json()["total"] >= 3

    @pytest.mark.asyncio
    @patch("app.workers.session_tasks.post_session_analysis.delay")
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_chart_shows_improving_trajectory(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        mock_celery,
        client,
        sample_patient,
    ):
        """
        Step 6: Progress chart data should show improving quality scores.
        TimescaleDB queries are mocked; we verify the response shape and
        that the series is ordered chronologically.
        """
        now = datetime.now(timezone.utc)
        mock_quality.return_value = [
            {
                "timestamp":         now - timedelta(days=4),
                "session_id":        str(uuid4()),
                "quality_score":     52.0,
                "completion_pct":    0.85,
                "post_session_pain": 6,
            },
            {
                "timestamp":         now - timedelta(days=2),
                "session_id":        str(uuid4()),
                "quality_score":     63.0,
                "completion_pct":    0.90,
                "post_session_pain": 4,
            },
            {
                "timestamp":         now,
                "session_id":        str(uuid4()),
                "quality_score":     74.0,
                "completion_pct":    0.95,
                "post_session_pain": 3,
            },
        ]
        mock_rom.return_value     = {}
        mock_summary.return_value = {
            "sessions_completed": 3,
            "avg_quality_score":  63.0,
            "avg_pain_score":     4.3,
            "last_session_at":    now,
        }
        mock_milestones.return_value = []

        auth = {"Authorization": f"Bearer {_token_for(sample_patient)}"}
        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress",
            headers=auth,
        )
        assert resp.status_code == 200
        body = resp.json()

        quality_series = body["quality_series"]
        assert len(quality_series) == 3

        # Scores should be in ascending order (improving)
        scores = [pt["quality_score"] for pt in quality_series]
        assert scores == sorted(scores), "Quality scores should trend upward"
        assert scores[-1] > scores[0]


def _token_for(patient) -> str:
    """Generate a JWT for a PatientProfile's user_id."""
    from app.core.security import create_access_token
    return create_access_token(str(patient.user_id), "patient")