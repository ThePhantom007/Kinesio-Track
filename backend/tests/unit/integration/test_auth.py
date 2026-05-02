"""
Integration tests for POST /api/v1/auth/* endpoints.

Tests the full HTTP request/response cycle against a real (test) DB with
per-test rollback.  Claude and Redis are mocked.
"""

from __future__ import annotations

import pytest


class TestRegister:

    @pytest.mark.asyncio
    async def test_register_patient_returns_201_and_tokens(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email":     "newpatient@test.local",
            "password":  "StrongPass1!",
            "full_name": "New Patient",
            "role":      "patient",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert "tokens" in body
        assert "access_token"  in body["tokens"]
        assert "refresh_token" in body["tokens"]
        assert body["user"]["role"] == "patient"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_returns_409(self, client):
        payload = {
            "email":     "dup@test.local",
            "password":  "StrongPass1!",
            "full_name": "First",
            "role":      "patient",
        }
        await client.post("/api/v1/auth/register", json=payload)
        resp = await client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "conflict"

    @pytest.mark.asyncio
    async def test_register_weak_password_returns_422(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email":     "weak@test.local",
            "password":  "abc",
            "full_name": "Weak",
            "role":      "patient",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_invalid_email_returns_422(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email":     "not-an-email",
            "password":  "StrongPass1!",
            "full_name": "Bad Email",
            "role":      "patient",
        })
        assert resp.status_code == 422


class TestLogin:

    @pytest.mark.asyncio
    async def test_login_valid_credentials_returns_tokens(self, client):
        # Register first
        await client.post("/api/v1/auth/register", json={
            "email":     "logintest@test.local",
            "password":  "StrongPass1!",
            "full_name": "Login Test",
            "role":      "patient",
        })
        resp = await client.post("/api/v1/auth/login", json={
            "email":    "logintest@test.local",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "tokens" in body
        assert body["user"]["email"] == "logintest@test.local"

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(self, client):
        await client.post("/api/v1/auth/register", json={
            "email":     "wrongpass@test.local",
            "password":  "StrongPass1!",
            "full_name": "Wrong Pass",
            "role":      "patient",
        })
        resp = await client.post("/api/v1/auth/login", json={
            "email":    "wrongpass@test.local",
            "password": "WrongPassword1!",
        })
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "authentication_error"

    @pytest.mark.asyncio
    async def test_login_unknown_email_returns_401(self, client):
        resp = await client.post("/api/v1/auth/login", json={
            "email":    "nobody@test.local",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_normalises_email_case(self, client):
        await client.post("/api/v1/auth/register", json={
            "email":     "casetest@test.local",
            "password":  "StrongPass1!",
            "full_name": "Case Test",
            "role":      "patient",
        })
        resp = await client.post("/api/v1/auth/login", json={
            "email":    "CASETEST@TEST.LOCAL",
            "password": "StrongPass1!",
        })
        assert resp.status_code == 200


class TestRefresh:

    @pytest.mark.asyncio
    async def test_refresh_valid_token_returns_new_tokens(self, client):
        reg_resp = await client.post("/api/v1/auth/register", json={
            "email":     "refreshtest@test.local",
            "password":  "StrongPass1!",
            "full_name": "Refresh Test",
            "role":      "patient",
        })
        refresh_token = reg_resp.json()["tokens"]["refresh_token"]

        resp = await client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token"  in body
        assert "refresh_token" in body

    @pytest.mark.asyncio
    async def test_refresh_invalid_token_returns_401(self, client):
        resp = await client.post("/api/v1/auth/refresh", json={
            "refresh_token": "not.a.valid.token",
        })
        assert resp.status_code == 401


class TestLogout:

    @pytest.mark.asyncio
    async def test_logout_returns_200(self, client):
        reg_resp = await client.post("/api/v1/auth/register", json={
            "email":     "logouttest@test.local",
            "password":  "StrongPass1!",
            "full_name": "Logout Test",
            "role":      "patient",
        })
        tokens = reg_resp.json()["tokens"]

        resp = await client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200


class TestMe:

    @pytest.mark.asyncio
    async def test_me_returns_user_profile(self, client, auth_headers):
        resp = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "email" in body
        assert "role"  in body

    @pytest.mark.asyncio
    async def test_me_without_token_returns_401(self, client):
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code == 401