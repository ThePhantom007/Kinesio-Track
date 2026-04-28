"""
Celery application instance and queue configuration.

Queues
------
  default         — general-purpose tasks (plan adaptation, notifications)
  video           — video processing tasks (CPU-intensive, separate pool)
  session         — post-session analysis tasks (time-sensitive)
  notifications   — push/email/webhook dispatch (I/O bound)

Workers are started with queue affinity so video processing never starves
the session analysis queue:

  # API / session worker (fast tasks)
  celery -A app.workers.celery_app worker -Q default,session,notifications

  # Video worker (CPU-intensive, can be scaled separately)
  celery -A app.workers.celery_app worker -Q video --concurrency 2

Beat schedule (periodic tasks) is defined in beat_schedule.py and registered
here at app creation time.
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

from app.core.config import settings

# ── App creation ──────────────────────────────────────────────────────────────

celery_app = Celery(
    "kinesio_track",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.workers.video_tasks",
        "app.workers.session_tasks",
        "app.workers.plan_tasks",
        "app.workers.notification_tasks",
        "app.workers.analytics_tasks",
        "app.workers.cleanup_tasks",
    ],
)

# ── Configuration ─────────────────────────────────────────────────────────────

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Result backend
    result_expires=3600,          # keep results for 1 h
    result_backend_transport_options={
        "retry_policy": {"timeout": 5.0},
    },
    # Task execution
    task_acks_late=True,          # ack after task completes, not before
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1, # fair dispatch — don't prefetch more than 1
    task_time_limit=600,          # hard kill after 10 min
    task_soft_time_limit=540,     # SoftTimeLimitExceeded raised at 9 min
    # Retries
    task_max_retries=3,
    task_default_retry_delay=60,  # seconds
    # Logging
    worker_hijack_root_logger=False,  # let structlog handle logging
    worker_redirect_stdouts=False,
)

# ── Named queues ──────────────────────────────────────────────────────────────

default_exchange = Exchange("default", type="direct")

celery_app.conf.task_queues = (
    Queue("default",       default_exchange, routing_key="default"),
    Queue("video",         default_exchange, routing_key="video"),
    Queue("session",       default_exchange, routing_key="session"),
    Queue("notifications", default_exchange, routing_key="notifications"),
)

celery_app.conf.task_default_queue    = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"

celery_app.conf.task_routes = {
    "app.workers.video_tasks.*":        {"queue": "video"},
    "app.workers.session_tasks.*":      {"queue": "session"},
    "app.workers.notification_tasks.*": {"queue": "notifications"},
    "app.workers.plan_tasks.*":         {"queue": "default"},
    "app.workers.analytics_tasks.*":    {"queue": "default"},
    "app.workers.cleanup_tasks.*":      {"queue": "default"},
}

# ── Beat schedule (imported from beat_schedule.py) ────────────────────────────

from app.workers.beat_schedule import BEAT_SCHEDULE  # noqa: E402
celery_app.conf.beat_schedule = BEAT_SCHEDULE