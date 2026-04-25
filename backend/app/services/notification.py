"""
Unified notification dispatcher — single interface for all outbound alerts.

Supported channels
------------------
  FCM push      Android app (Firebase Cloud Messaging)
  Web Push      Browser notifications (VAPID)
  Email         SMTP via aiosmtplib
  Webhook       Clinician-configured HTTPS endpoint (red-flag alerts)

Each send_*() method is fire-and-forget: failures are logged but never
re-raised, because a notification failure must never block the primary flow.

Usage::

    notif = NotificationService()
    await notif.send_red_flag_alert(clinician=clinician, event=red_flag_event)
    await notif.send_session_summary(patient=patient, session=session)
    await notif.send_milestone(patient=patient, label="Completed Phase 1!")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiohttp
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.core.config import settings
from app.core.logging import get_logger
from app.models.clinician import ClinicianProfile
from app.models.patient import PatientProfile
from app.models.red_flag import RedFlagEvent, RedFlagSeverity
from app.models.session import ExerciseSession

log = get_logger(__name__)

_WEBHOOK_TIMEOUT_SECONDS = 5


class NotificationService:

    # ═══════════════════════════════════════════════════════════════════════════
    # Red-flag alerts
    # ═══════════════════════════════════════════════════════════════════════════

    async def send_red_flag_alert(
        self,
        *,
        clinician: ClinicianProfile | None,
        event: RedFlagEvent,
        patient_name: str | None = None,
    ) -> None:
        """
        Notify the assigned clinician of a red-flag escalation via all
        configured channels (webhook + email).

        Args:
            clinician:    Assigned ClinicianProfile, or None if unassigned.
            event:        RedFlagEvent ORM object just written.
            patient_name: Optional display name for the notification body.
        """
        if clinician is None:
            log.warning(
                "red_flag_no_clinician",
                event_id=str(event.id),
                severity=event.severity.value,
            )
            return

        severity_emoji = {"warn": "⚠️", "stop": "🛑", "seek_care": "🚨"}.get(
            event.severity.value, "⚠️"
        )
        subject = (
            f"{severity_emoji} Kinesio-Track Red Flag: {event.trigger_type.value.replace('_', ' ').title()} "
            f"— {patient_name or 'Patient'}"
        )
        body = self._format_red_flag_body(event, patient_name)

        # Webhook (fastest — real-time dashboard integration)
        if clinician.webhook_url:
            await self._post_webhook(
                url=clinician.webhook_url,
                payload={
                    "event_type":     "red_flag",
                    "event_id":       str(event.id),
                    "severity":       event.severity.value,
                    "trigger_type":   event.trigger_type.value,
                    "immediate_action": event.immediate_action,
                    "clinician_note": event.clinician_note,
                    "patient_name":   patient_name,
                    "timestamp":      datetime.now(timezone.utc).isoformat(),
                },
            )

        # Email
        if clinician.email_alerts_enabled:
            clinician_user_email = getattr(clinician.user, "email", None)
            if clinician_user_email:
                await self._send_email(
                    to=clinician_user_email,
                    subject=subject,
                    body=body,
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # Session notifications (patient-facing)
    # ═══════════════════════════════════════════════════════════════════════════

    async def send_session_summary(
        self,
        *,
        patient: PatientProfile,
        session: ExerciseSession,
        summary_text: str,
    ) -> None:
        """Send the post-session summary to the patient via FCM or Web Push."""
        body = (
            f"Great work! Here's your session summary:\n\n{summary_text}"
            if summary_text
            else "Session complete — well done!"
        )
        await self._send_to_patient(
            patient=patient,
            title="Session Complete 🎉",
            body=body,
            data={"session_id": str(session.id), "type": "session_summary"},
        )

    async def send_session_reminder(
        self,
        *,
        patient: PatientProfile,
        exercise_name: str,
    ) -> None:
        """Daily reminder to complete today's exercise session."""
        await self._send_to_patient(
            patient=patient,
            title="Time for your physio 💪",
            body=f"Your next exercise is ready: {exercise_name}. Tap to start.",
            data={"type": "session_reminder"},
        )

    async def send_milestone(
        self,
        *,
        patient: PatientProfile,
        label: str,
    ) -> None:
        """Notify a patient of a recovery milestone."""
        await self._send_to_patient(
            patient=patient,
            title="Milestone reached! 🏆",
            body=label,
            data={"type": "milestone"},
        )

    async def send_missed_session_alert(
        self,
        *,
        patient: PatientProfile,
        days_missed: int,
    ) -> None:
        """Alert a patient who hasn't done a session in several days."""
        await self._send_to_patient(
            patient=patient,
            title="We miss you 👋",
            body=(
                f"You haven't done a session in {days_missed} day(s). "
                "Consistency is key to recovery — tap to get back on track!"
            ),
            data={"type": "missed_session"},
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Delivery methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def _send_to_patient(
        self,
        *,
        patient: PatientProfile,
        title: str,
        body: str,
        data: dict[str, str] | None = None,
    ) -> None:
        """Route notification to the appropriate channel for this patient."""
        if patient.fcm_token:
            await self._send_fcm(
                token=patient.fcm_token,
                title=title,
                body=body,
                data=data or {},
            )
        elif patient.web_push_subscription:
            await self._send_web_push(
                subscription=patient.web_push_subscription,
                title=title,
                body=body,
            )
        else:
            log.debug(
                "no_push_channel",
                patient_id=str(patient.id),
                title=title,
            )

    async def _send_fcm(
        self,
        *,
        token: str,
        title: str,
        body: str,
        data: dict[str, str],
    ) -> None:
        """
        Send an FCM v1 notification.
        Requires FCM_SERVER_KEY in settings (OAuth 2.0 bearer token for v1 API).
        """
        if not settings.FCM_SERVER_KEY:
            log.debug("fcm_not_configured")
            return

        payload = {
            "message": {
                "token": token,
                "notification": {"title": title, "body": body},
                "data": data,
                "android": {"priority": "high"},
            }
        }
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    "https://fcm.googleapis.com/v1/projects/kinesio-track/messages:send",
                    headers={
                        "Authorization": f"Bearer {settings.FCM_SERVER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                if resp.status != 200:
                    body_text = await resp.text()
                    log.warning("fcm_send_failed", status=resp.status, response=body_text[:200])
                else:
                    log.debug("fcm_sent", title=title)
        except Exception as exc:
            log.warning("fcm_send_error", error=str(exc))

    async def _send_web_push(
        self,
        *,
        subscription: dict[str, Any],
        title: str,
        body: str,
    ) -> None:
        """
        Send a Web Push notification using the pywebpush library.
        Falls back silently if pywebpush is not installed.
        """
        try:
            from pywebpush import webpush, WebPushException
            webpush(
                subscription_info=subscription,
                data=json.dumps({"title": title, "body": body}),
                vapid_private_key=settings.SECRET_KEY,
                vapid_claims={"sub": f"mailto:{settings.EMAIL_FROM}"},
            )
            log.debug("web_push_sent", title=title)
        except ImportError:
            log.debug("pywebpush_not_installed")
        except Exception as exc:
            log.warning("web_push_error", error=str(exc))

    async def _send_email(
        self,
        *,
        to: str,
        subject: str,
        body: str,
    ) -> None:
        """Send a plain-text email via SMTP."""
        if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
            log.debug("smtp_not_configured")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = settings.EMAIL_FROM
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain"))

        try:
            await aiosmtplib.send(
                msg,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASSWORD,
                start_tls=True,
            )
            log.debug("email_sent", to=to, subject=subject[:60])
        except Exception as exc:
            log.warning("email_send_error", error=str(exc), to=to)

    async def _post_webhook(
        self,
        *,
        url: str,
        payload: dict[str, Any],
    ) -> None:
        """POST a JSON payload to a clinician's webhook URL."""
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_SECONDS),
                )
                if resp.status not in (200, 201, 202, 204):
                    log.warning(
                        "webhook_non_2xx",
                        url=url[:80],
                        status=resp.status,
                    )
                else:
                    log.debug("webhook_delivered", url=url[:80])
        except Exception as exc:
            log.warning("webhook_delivery_error", url=url[:80], error=str(exc))

    # ── Formatters ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_red_flag_body(event: RedFlagEvent, patient_name: str | None) -> str:
        name = patient_name or "Your patient"
        severity_label = event.severity.value.replace("_", " ").upper()
        return (
            f"RED FLAG ALERT — {severity_label}\n"
            f"Patient: {name}\n"
            f"Trigger: {event.trigger_type.value.replace('_', ' ').title()}\n\n"
            f"Patient message sent:\n{event.immediate_action}\n\n"
            f"Clinical note:\n{event.clinician_note}\n\n"
            f"Session recommendation: {event.session_recommendation or 'N/A'}\n\n"
            f"Please log in to Kinesio-Track to review and acknowledge this alert.\n"
        )