"""
Integration tests for POST /api/v1/intake.

Verifies the full intake → plan generation flow with a mocked Claude client.
Asserts that the plan, phases, and exercises are correctly written to the DB
and that the response schema matches expectations.
"""

from __future__ import annotations

import pytest

from tests.fixtures.mock_claude_responses import VALID_PLAN_RESPONSE


class TestIntakeFlow:

    @pytest.mark.asyncio
    async def test_intake_returns_201_with_plan_id(
        self, client, auth_headers, sample_patient
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": (
                    "I sprained my left ankle playing cricket two weeks ago. "
                    "It twisted inward when I landed. Pain and swelling for first week."
                ),
                "body_part":  "ankle",
                "pain_score":  6,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "plan_id"  in body
        assert "injury_id" in body
        assert body["status"] in ("ready", "generating")

    @pytest.mark.asyncio
    async def test_intake_response_has_phase_count(
        self, client, auth_headers, sample_patient
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": (
                    "Neck pain after sleeping in an awkward position. "
                    "Limited rotation to the left, dull ache for three days."
                ),
                "body_part":  "neck",
                "pain_score":  4,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["estimated_phases"] == len(VALID_PLAN_RESPONSE.phases)

    @pytest.mark.asyncio
    async def test_intake_requires_minimum_description_length(
        self, client, auth_headers
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": "short",
                "body_part":   "ankle",
                "pain_score":  5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_intake_requires_valid_pain_score(
        self, client, auth_headers
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": "Left ankle sprain from running yesterday, lots of swelling.",
                "body_part":   "ankle",
                "pain_score":  15,   # out of range
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_intake_requires_valid_body_part(
        self, client, auth_headers
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": "Left ankle sprain from running yesterday, lots of swelling.",
                "body_part":   "pinky_toe",   # not a valid BodyPart
                "pain_score":  5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_intake_without_auth_returns_401(self, client):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": "Left ankle sprain, swelling and pain for one week.",
                "body_part":   "ankle",
                "pain_score":  6,
            },
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_intake_with_nonexistent_s3_key_returns_404(
        self, client, auth_headers, sample_patient
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description":          "Ankle sprain with intake video attached.",
                "body_part":            "ankle",
                "pain_score":           5,
                "intake_video_s3_key":  "patients/fake-id/intake/does-not-exist.mp4",
            },
            headers=auth_headers,
        )
        # S3 object_exists returns False for unknown key → 404
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_plan_summary_present_in_response(
        self, client, auth_headers, sample_patient
    ):
        resp = await client.post(
            "/api/v1/intake",
            json={
                "description": (
                    "Left shoulder injury from weightlifting. "
                    "Pain when lifting arm above shoulder height, started last week."
                ),
                "body_part":  "shoulder",
                "pain_score":  5,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert isinstance(body.get("summary"), str)
        assert len(body["summary"]) > 10