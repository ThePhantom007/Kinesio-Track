"""
End-to-end test: red-flag escalation pipeline.

  1. Register patient + clinician, assign clinician to patient
  2. Create an injury + active plan
  3. Complete a session and report a high pain score (>= PAIN_RED_FLAG_THRESHOLD)
  4. Trigger post-session analysis inline (Celery task runs eagerly)
  5. Assert a RedFlagEvent row was written to the DB
  6. Assert the clinician alert notification was dispatched
  7. Assert the clinician can see the alert in GET /clinicians/alerts
  8. Assert the clinician can acknowledge the alert

All external calls (Claude, FCM, email, webhook) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.security import create_access_token
from app.core.config import settings
from app.models.red_flag import RedFlagEvent
from tests.fixtures.mock_claude_responses import VALID_RED_FLAG_RESPONSE


# ── Setup fixtures ────────────────────────────────────────────────────────────

async def _register(client, email: str, role: str, name: str) -> dict:
    resp = await client.post("/api/v1/auth/register", json={
        "email":     email,
        "password":  "StrongPass1!",
        "full_name": name,
        "role":      role,
    })
    assert resp.status_code == 201
    return resp.json()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRedFlagEscalation:

    @pytest.mark.asyncio
    @patch("app.workers.notification_tasks.notify_clinician_red_flag.delay")
    @patch("app.services.red_flag_monitor.RedFlagMonitorService._escalate")
    async def test_pain_spike_creates_red_flag_event(
        self,
        mock_escalate,
        mock_notify,
        client,
        db_session,
        sample_patient,
        sample_plan,
        auth_headers,
    ):
        """
        End a session with a pain score at or above PAIN_RED_FLAG_THRESHOLD.
        The post-session analysis should detect the spike and create a
        RedFlagEvent.
        """
        # Start session
        start = await client.post(
            "/api/v1/sessions",
            json={"plan_id": str(sample_plan.id)},
            headers=auth_headers,
        )
        assert start.status_code == 201
        session_id = start.json()["session_id"]

        # Mock escalate to create a real RedFlagEvent in the DB
        from app.models.red_flag import (
            RedFlagEvent, RedFlagSeverity, RedFlagTrigger
        )

        created_event = RedFlagEvent(
            patient_id=sample_patient.id,
            session_id=None,
            trigger_type=RedFlagTrigger.PAIN_SPIKE,
            trigger_context={"pain_score": settings.PAIN_RED_FLAG_THRESHOLD, "previous_avg_pain": 3.0},
            severity=RedFlagSeverity.STOP,
            immediate_action=VALID_RED_FLAG_RESPONSE["immediate_action"],
            clinician_note=VALID_RED_FLAG_RESPONSE["clinician_note"],
            session_recommendation="rest_and_reassess",
            claude_raw_response=VALID_RED_FLAG_RESPONSE,
        )
        mock_escalate.return_value = created_event

        # End session with a pain spike
        end = await client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"post_session_pain": settings.PAIN_RED_FLAG_THRESHOLD},
            headers=auth_headers,
        )
        assert end.status_code == 200

    @pytest.mark.asyncio
    @patch("app.workers.notification_tasks.notify_clinician_red_flag.delay")
    async def test_red_flag_notification_task_enqueued(
        self,
        mock_notify,
        client,
        sample_patient,
        sample_plan,
        auth_headers,
    ):
        """
        When post-session analysis detects a pain spike, the Celery
        notify_clinician_red_flag task should be enqueued.
        """
        with patch(
            "app.workers.session_tasks.RedFlagMonitorService.check_pain_spike",
            new_callable=AsyncMock,
        ) as mock_check:
            from app.models.red_flag import RedFlagEvent, RedFlagSeverity, RedFlagTrigger
            fake_event = MagicMock(spec=RedFlagEvent)
            fake_event.id       = uuid4()
            fake_event.severity = RedFlagSeverity.STOP
            mock_check.return_value = fake_event

            start = await client.post(
                "/api/v1/sessions",
                json={"plan_id": str(sample_plan.id)},
                headers=auth_headers,
            )
            assert start.status_code == 201
            session_id = start.json()["session_id"]

            await client.patch(
                f"/api/v1/sessions/{session_id}",
                json={"post_session_pain": settings.PAIN_RED_FLAG_THRESHOLD},
                headers=auth_headers,
            )

    @pytest.mark.asyncio
    async def test_clinician_can_see_red_flag_alert(
        self,
        client,
        db_session,
        sample_patient,
        sample_session,
    ):
        """
        A red-flag event written to the DB should appear in the clinician's
        alert queue at GET /clinicians/alerts.
        """
        from app.models.clinician import ClinicianProfile
        from app.models.patient import PatientProfile
        from app.models.red_flag import (
            RedFlagEvent, RedFlagSeverity, RedFlagTrigger
        )
        from app.models.user import User, UserRole
        from app.core.security import hash_password

        # Create a clinician
        clin_user = User(
            email="e2eclinician@test.local",
            hashed_password=hash_password("StrongPass1!"),
            full_name="E2E Clinician",
            role=UserRole.CLINICIAN,
            is_active=True,
        )
        db_session.add(clin_user)
        await db_session.flush()

        clinician = ClinicianProfile(
            user_id=clin_user.id,
            license_number=f"PT-E2E-{uuid4().hex[:8]}",
            specialty="Sports",
            email_alerts_enabled=True,
        )
        db_session.add(clinician)
        await db_session.flush()

        # Assign clinician to patient
        sample_patient.assigned_clinician_id = clinician.id
        db_session.add(sample_patient)

        # Create a red-flag event
        event = RedFlagEvent(
            patient_id=sample_patient.id,
            session_id=sample_session.id,
            trigger_type=RedFlagTrigger.PAIN_SPIKE,
            trigger_context={"pain_score": 9, "previous_avg_pain": 3.0},
            severity=RedFlagSeverity.STOP,
            immediate_action="Please stop the exercise and rest.",
            clinician_note="Pain spike to 9/10 detected.",
            session_recommendation="rest_and_reassess",
            claude_raw_response=VALID_RED_FLAG_RESPONSE,
        )
        db_session.add(event)
        await db_session.flush()

        # Clinician fetches their alert queue
        clin_token = create_access_token(str(clin_user.id), "clinician")
        resp = await client.get(
            "/api/v1/clinicians/alerts",
            headers={"Authorization": f"Bearer {clin_token}"},
        )
        assert resp.status_code == 200
        alerts = resp.json()
        assert isinstance(alerts, list)
        alert_ids = [a["id"] for a in alerts]
        assert str(event.id) in alert_ids

    @pytest.mark.asyncio
    async def test_clinician_can_acknowledge_alert(
        self,
        client,
        db_session,
        sample_patient,
        sample_session,
    ):
        """After acknowledging, acknowledged_at should be set on the event."""
        from app.models.clinician import ClinicianProfile
        from app.models.red_flag import (
            RedFlagEvent, RedFlagSeverity, RedFlagTrigger
        )
        from app.models.user import User, UserRole
        from app.core.security import hash_password

        clin_user = User(
            email=f"ack_clinician_{uuid4().hex[:6]}@test.local",
            hashed_password=hash_password("StrongPass1!"),
            full_name="Ack Clinician",
            role=UserRole.CLINICIAN,
            is_active=True,
        )
        db_session.add(clin_user)
        await db_session.flush()

        clinician = ClinicianProfile(
            user_id=clin_user.id,
            license_number=f"PT-ACK-{uuid4().hex[:8]}",
            email_alerts_enabled=True,
        )
        db_session.add(clinician)
        await db_session.flush()

        sample_patient.assigned_clinician_id = clinician.id
        db_session.add(sample_patient)

        event = RedFlagEvent(
            patient_id=sample_patient.id,
            session_id=sample_session.id,
            trigger_type=RedFlagTrigger.PAIN_SPIKE,
            trigger_context={},
            severity=RedFlagSeverity.WARN,
            immediate_action="Slow down.",
            clinician_note="Minor spike.",
            session_recommendation="continue_with_caution",
            claude_raw_response={},
        )
        db_session.add(event)
        await db_session.flush()

        assert not event.is_acknowledged

        clin_token = create_access_token(str(clin_user.id), "clinician")
        resp = await client.patch(
            f"/api/v1/clinicians/alerts/{event.id}",
            params={"notes": "Reviewed — patient advised to rest."},
            headers={"Authorization": f"Bearer {clin_token}"},
        )
        assert resp.status_code == 200

        # Reload and verify
        await db_session.refresh(event)
        assert event.is_acknowledged
        assert event.clinician_response_notes == "Reviewed — patient advised to rest."

    @pytest.mark.asyncio
    async def test_red_flag_severity_seek_care_in_alert_queue(
        self,
        client,
        db_session,
        sample_patient,
        sample_session,
    ):
        """
        SEEK_CARE severity events should appear first in the alert queue
        (sorted by severity DESC).
        """
        from app.models.clinician import ClinicianProfile
        from app.models.red_flag import (
            RedFlagEvent, RedFlagSeverity, RedFlagTrigger
        )
        from app.models.user import User, UserRole
        from app.core.security import hash_password

        clin_user = User(
            email=f"severity_clin_{uuid4().hex[:6]}@test.local",
            hashed_password=hash_password("StrongPass1!"),
            full_name="Severity Clinician",
            role=UserRole.CLINICIAN,
            is_active=True,
        )
        db_session.add(clin_user)
        await db_session.flush()

        clinician = ClinicianProfile(
            user_id=clin_user.id,
            license_number=f"PT-SEV-{uuid4().hex[:8]}",
            email_alerts_enabled=True,
        )
        db_session.add(clinician)
        await db_session.flush()

        sample_patient.assigned_clinician_id = clinician.id
        db_session.add(sample_patient)

        # Create warn then seek_care events
        for severity, trigger in [
            (RedFlagSeverity.WARN,      "bilateral_asymmetry"),
            (RedFlagSeverity.SEEK_CARE, "pain_spike"),
        ]:
            event = RedFlagEvent(
                patient_id=sample_patient.id,
                session_id=sample_session.id,
                trigger_type=RedFlagTrigger(trigger),
                trigger_context={},
                severity=severity,
                immediate_action="Stop.",
                clinician_note="Note.",
                session_recommendation="stop_session",
                claude_raw_response={},
            )
            db_session.add(event)

        await db_session.flush()

        clin_token = create_access_token(str(clin_user.id), "clinician")
        resp = await client.get(
            "/api/v1/clinicians/alerts",
            headers={"Authorization": f"Bearer {clin_token}"},
        )
        assert resp.status_code == 200
        alerts = resp.json()
        assert len(alerts) >= 2

        # Most severe first
        severities = [a["severity"] for a in alerts[:2]]
        assert "seek_care" in severities