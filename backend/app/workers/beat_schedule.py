"""
Celery Beat periodic task schedule.

All times are UTC.  crontab() uses standard cron syntax.

Tasks
-----
  daily_session_reminders       08:00 UTC daily
    Send push/email reminders to patients who have not done a session today.

  daily_missed_session_check    20:00 UTC daily
    Flag patients who have missed 3+ consecutive scheduled sessions.

  weekly_progress_digest        09:00 UTC every Monday
    Send each active patient a weekly summary of their progress.

  daily_recovery_eta_refresh    03:00 UTC daily
    Recompute recovery ETAs for all active patients using the latest session
    data.  Results are cached so dashboard loads are fast.

  daily_cleanup                 02:00 UTC daily
    Purge expired Redis keys, delete orphaned S3 uploads, archive old sessions.

  daily_budget_check            07:00 UTC daily
    Check monthly Claude API spend against MONTHLY_TOKEN_BUDGET_USD and
    send an alert if usage exceeds 80% or 100% of the budget.
"""

from __future__ import annotations

from celery.schedules import crontab

BEAT_SCHEDULE: dict = {
    # ── Patient reminders ──────────────────────────────────────────────────────
    "daily-session-reminders": {
        "task":     "app.workers.notification_tasks.send_daily_session_reminders",
        "schedule": crontab(hour=8, minute=0),
        "options":  {"queue": "notifications"},
    },
    "daily-missed-session-check": {
        "task":     "app.workers.notification_tasks.check_missed_sessions",
        "schedule": crontab(hour=20, minute=0),
        "options":  {"queue": "notifications"},
    },
    "weekly-progress-digest": {
        "task":     "app.workers.notification_tasks.send_weekly_progress_digest",
        "schedule": crontab(hour=9, minute=0, day_of_week=1),   # Monday
        "options":  {"queue": "notifications"},
    },

    # ── Analytics ──────────────────────────────────────────────────────────────
    "daily-recovery-eta-refresh": {
        "task":     "app.workers.analytics_tasks.refresh_all_recovery_etas",
        "schedule": crontab(hour=3, minute=0),
        "options":  {"queue": "default"},
    },
    "daily-timescale-aggregate-refresh": {
        "task":     "app.workers.analytics_tasks.refresh_timescale_aggregates",
        "schedule": crontab(hour=3, minute=30),
        "options":  {"queue": "default"},
    },

    # ── Maintenance ────────────────────────────────────────────────────────────
    "daily-cleanup": {
        "task":     "app.workers.cleanup_tasks.daily_cleanup",
        "schedule": crontab(hour=2, minute=0),
        "options":  {"queue": "default"},
    },
    "daily-budget-check": {
        "task":     "app.workers.analytics_tasks.check_claude_budget",
        "schedule": crontab(hour=7, minute=0),
        "options":  {"queue": "default"},
    },
}