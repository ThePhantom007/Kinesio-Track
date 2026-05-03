"""
Integration tests for the progress dashboard endpoints:
  GET /api/v1/patients/{id}/progress
  GET /api/v1/patients/{id}/progress/recovery-eta

These tests mock the TimescaleDB query functions to avoid requiring a
live TimescaleDB instance, while still verifying the full HTTP → service
→ response schema pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_quality_points(n: int = 5) -> list[dict]:
    return [
        {
            "timestamp":        datetime(2026, 4, i + 1, tzinfo=timezone.utc),
            "session_id":       str(uuid4()),
            "quality_score":    50.0 + i * 5,
            "completion_pct":   0.85,
            "post_session_pain": max(6 - i, 1),
        }
        for i in range(n)
    ]


def _make_rom_points(n: int = 5, joint: str = "left_ankle") -> list[dict]:
    return [
        {
            "timestamp":        datetime(2026, 4, i + 1, tzinfo=timezone.utc),
            "session_id":       str(uuid4()),
            "joint":            joint,
            "avg_angle_deg":    18.0 + i * 1.5,
            "peak_angle_deg":   20.0 + i * 1.5,
            "avg_quality_score": 55.0 + i * 4,
        }
        for i in range(n)
    ]


def _make_summary() -> dict:
    return {
        "sessions_completed": 5,
        "avg_quality_score":  67.5,
        "avg_pain_score":     3.8,
        "last_session_at":    datetime(2026, 4, 10, tzinfo=timezone.utc),
    }


# ── Progress endpoint ─────────────────────────────────────────────────────────

class TestProgressEndpoint:

    @pytest.mark.asyncio
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_returns_200(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        mock_quality.return_value    = _make_quality_points()
        mock_rom.return_value        = {"left_ankle": _make_rom_points()}
        mock_summary.return_value    = _make_summary()
        mock_milestones.return_value = []

        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_response_schema(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        mock_quality.return_value    = _make_quality_points()
        mock_rom.return_value        = {"left_ankle": _make_rom_points()}
        mock_summary.return_value    = _make_summary()
        mock_milestones.return_value = []

        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress",
            headers=auth_headers,
        )
        body = resp.json()
        assert "patient_id"      in body
        assert "quality_series"  in body
        assert "rom_series"      in body
        assert "sessions_completed" in body
        assert isinstance(body["quality_series"], list)
        assert isinstance(body["rom_series"],     list)

    @pytest.mark.asyncio
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_quality_points_populated(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        mock_quality.return_value    = _make_quality_points(5)
        mock_rom.return_value        = {}
        mock_summary.return_value    = _make_summary()
        mock_milestones.return_value = []

        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress",
            headers=auth_headers,
        )
        body = resp.json()
        assert len(body["quality_series"]) == 5
        for pt in body["quality_series"]:
            assert "quality_score" in pt
            assert "timestamp"     in pt

    @pytest.mark.asyncio
    async def test_progress_wrong_patient_returns_403(
        self, client, auth_headers
    ):
        other_patient_id = uuid4()
        resp = await client.get(
            f"/api/v1/patients/{other_patient_id}/progress",
            headers=auth_headers,
        )
        # Patient can only see own progress
        assert resp.status_code in (403, 404)

    @pytest.mark.asyncio
    async def test_progress_requires_auth(self, client, sample_patient):
        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress"
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_granularity_param_accepted(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        mock_quality.return_value    = []
        mock_rom.return_value        = {}
        mock_summary.return_value    = _make_summary()
        mock_milestones.return_value = []

        for granularity in ("session", "daily", "weekly"):
            resp = await client.get(
                f"/api/v1/patients/{sample_patient.id}/progress?granularity={granularity}",
                headers=auth_headers,
            )
            assert resp.status_code == 200, f"Failed for granularity={granularity}"
            assert resp.json()["granularity"] == granularity

    @pytest.mark.asyncio
    @patch("app.api.v1.progress.quality_score_series", new_callable=AsyncMock)
    @patch("app.api.v1.progress.rom_series_all_joints",  new_callable=AsyncMock)
    @patch("app.api.v1.progress.progress_summary",       new_callable=AsyncMock)
    @patch("app.api.v1.progress.get_milestones",         new_callable=AsyncMock)
    async def test_progress_milestones_included(
        self,
        mock_milestones,
        mock_summary,
        mock_rom,
        mock_quality,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        mock_quality.return_value = []
        mock_rom.return_value     = {}
        mock_summary.return_value = _make_summary()
        mock_milestones.return_value = [
            {
                "milestone_type": "consecutive_sessions",
                "label":          "5 sessions completed!",
                "achieved_at":    datetime(2026, 4, 10, tzinfo=timezone.utc),
                "value":          5.0,
            }
        ]

        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress",
            headers=auth_headers,
        )
        body = resp.json()
        assert len(body["milestones"]) == 1
        assert body["milestones"][0]["milestone_type"] == "consecutive_sessions"


# ── Recovery ETA endpoint ─────────────────────────────────────────────────────

class TestRecoveryETA:

    @pytest.mark.asyncio
    @patch(
        "app.api.v1.progress.RecoveryForecasterService.forecast",
        new_callable=AsyncMock,
    )
    async def test_recovery_eta_returns_200(
        self,
        mock_forecast,
        client,
        auth_headers,
        sample_patient,
        sample_plan,
    ):
        from app.schemas.progress import RecoveryForecast
        mock_forecast.return_value = RecoveryForecast(
            estimated_recovery_date=None,
            estimated_days_remaining=None,
            confidence="low",
            trend="improving",
            sessions_analysed=2,
            ai_narrative=None,
            slope_per_session=None,
        )

        resp = await client.get(
            f"/api/v1/patients/{sample_patient.id}/progress/recovery-eta",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_recovery_eta_schema(
        self, client, auth_headers, sample_patient, sample_plan
    ):
        from unittest.mock import patch, AsyncMock
        from app.schemas.progress import RecoveryForecast

        with patch(
            "app.services.recovery_forecaster.RecoveryForecasterService.forecast",
            new_callable=AsyncMock,
            return_value=RecoveryForecast(
                estimated_recovery_date=None,
                estimated_days_remaining=None,
                confidence="low",
                trend="plateauing",
                sessions_analysed=1,
                ai_narrative=None,
                slope_per_session=None,
            ),
        ):
            resp = await client.get(
                f"/api/v1/patients/{sample_patient.id}/progress/recovery-eta",
                headers=auth_headers,
            )

        body = resp.json()
        assert "confidence"  in body
        assert "trend"       in body
        assert body["confidence"] in ("low", "moderate", "high")
        assert body["trend"]      in ("improving", "plateauing", "regressing")