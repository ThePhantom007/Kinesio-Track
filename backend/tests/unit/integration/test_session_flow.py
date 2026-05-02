"""
Integration tests for session lifecycle:
  POST  /api/v1/sessions              — start
  PATCH /api/v1/sessions/{id}         — end
  GET   /api/v1/sessions/{id}/summary — fetch summary
  GET   /api/v1/sessions              — list history
"""

from __future__ import annotations

import pytest


class TestStartSession:

    @pytest.mark.asyncio
    async def test_start_session_returns_201_with_ws_url(
        self, client, auth_headers, sample_plan, sample_patient
    ):
        resp = await client.post(
            "/api/v1/sessions",
            json={"plan_id": str(sample_plan.id)},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "session_id"    in body
        assert "websocket_url" in body
        assert "ws" in body["websocket_url"].lower()

    @pytest.mark.asyncio
    async def test_start_session_returns_first_exercise(
        self, client, auth_headers, sample_plan, sample_patient
    ):
        resp = await client.post(
            "/api/v1/sessions",
            json={"plan_id": str(sample_plan.id)},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "first_exercise" in body
        ex = body["first_exercise"]
        assert "name" in ex
        assert "sets" in ex
        assert "reps" in ex

    @pytest.mark.asyncio
    async def test_start_session_with_nonexistent_plan_returns_404(
        self, client, auth_headers
    ):
        import uuid
        resp = await client.post(
            "/api/v1/sessions",
            json={"plan_id": str(uuid.uuid4())},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_start_session_requires_auth(self, client, sample_plan):
        resp = await client.post(
            "/api/v1/sessions",
            json={"plan_id": str(sample_plan.id)},
        )
        assert resp.status_code == 401


class TestEndSession:

    @pytest.mark.asyncio
    async def test_end_session_returns_200_with_summary(
        self, client, auth_headers, sample_session, sample_patient
    ):
        resp = await client.patch(
            f"/api/v1/sessions/{sample_session.id}",
            json={"post_session_pain": 4},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "status"  in body

    @pytest.mark.asyncio
    async def test_end_session_pain_score_out_of_range_returns_422(
        self, client, auth_headers, sample_session
    ):
        resp = await client.patch(
            f"/api/v1/sessions/{sample_session.id}",
            json={"post_session_pain": 15},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_end_nonexistent_session_returns_404(
        self, client, auth_headers
    ):
        import uuid
        resp = await client.patch(
            f"/api/v1/sessions/{uuid.uuid4()}",
            json={"post_session_pain": 3},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestGetSessionSummary:

    @pytest.mark.asyncio
    async def test_get_summary_returns_200(
        self, client, auth_headers, sample_session
    ):
        resp = await client.get(
            f"/api/v1/sessions/{sample_session.id}/summary",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(sample_session.id)
        assert "metrics" in body

    @pytest.mark.asyncio
    async def test_summary_contains_expected_metric_fields(
        self, client, auth_headers, sample_session
    ):
        resp = await client.get(
            f"/api/v1/sessions/{sample_session.id}/summary",
            headers=auth_headers,
        )
        metrics = resp.json()["metrics"]
        for field in ("avg_quality_score", "completion_pct", "post_session_pain"):
            assert field in metrics

    @pytest.mark.asyncio
    async def test_get_summary_requires_auth(self, client, sample_session):
        resp = await client.get(f"/api/v1/sessions/{sample_session.id}/summary")
        assert resp.status_code == 401


class TestSessionHistory:

    @pytest.mark.asyncio
    async def test_list_sessions_returns_200(
        self, client, auth_headers, sample_session
    ):
        resp = await client.get("/api/v1/sessions", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "sessions" in body
        assert isinstance(body["sessions"], list)

    @pytest.mark.asyncio
    async def test_list_sessions_contains_sample_session(
        self, client, auth_headers, sample_session
    ):
        resp = await client.get("/api/v1/sessions", headers=auth_headers)
        ids  = [s["id"] for s in resp.json()["sessions"]]
        assert str(sample_session.id) in ids

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_plan(
        self, client, auth_headers, sample_session, sample_plan
    ):
        resp = await client.get(
            f"/api/v1/sessions?plan_id={sample_plan.id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert all(
            s.get("plan_id") == str(sample_plan.id) or "plan_id" not in s
            for s in body["sessions"]
        )

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(
        self, client, auth_headers, sample_session
    ):
        resp = await client.get(
            "/api/v1/sessions?limit=1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["sessions"]) <= 1
        assert "has_more" in body