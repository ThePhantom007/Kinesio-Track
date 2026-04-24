"""
Structured JSON logging via structlog.

Usage anywhere in the codebase:
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("session_started", session_id=str(session_id), patient_id=str(patient_id))

Context binding — FastAPI middleware automatically binds request_id and
user_id to the structlog context for the duration of each request, so all
log entries emitted during that request carry those fields without any
explicit passing.

Production output: one JSON object per line, suitable for Datadog / CloudWatch.
Development output: human-readable coloured console output.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import settings


# ── Custom processors ──────────────────────────────────────────────────────────

def _add_service_name(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Stamp every log entry with the service name and version."""
    event_dict.setdefault("service", settings.APP_NAME)
    event_dict.setdefault("version", settings.APP_VERSION)
    event_dict.setdefault("environment", settings.ENVIRONMENT)
    return event_dict


def _drop_color_message(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Uvicorn injects a pre-formatted 'color_message' key that pollutes JSON
    output.  Remove it before serialising.
    """
    event_dict.pop("color_message", None)
    return event_dict


# ── Setup ──────────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    """
    Call once at application startup (in ``app/main.py`` lifespan).
    Subsequent calls are idempotent.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_service_name,
        _drop_color_message,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        # Flat JSON — one line per event, easy to ingest in log aggregators.
        renderer = structlog.processors.JSONRenderer()
        log_level = logging.INFO
    else:
        # Pretty-printed, colour-coded output for local development.
        renderer = structlog.dev.ConsoleRenderer(colors=True)
        log_level = logging.DEBUG

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Clear any handlers configured by libraries before us.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Public factory ─────────────────────────────────────────────────────────────

def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to *name* (typically ``__name__``).

    Example::

        log = get_logger(__name__)
        log.info("plan_generated", plan_id=str(plan.id), patient_id=str(patient.id))
    """
    return structlog.get_logger(name)


# ── Context helpers ────────────────────────────────────────────────────────────

def bind_request_context(
    *,
    request_id: str,
    user_id: str | None = None,
    patient_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """
    Bind fields to the structlog context-vars context for the current
    async task (request).  The middleware calls this; application code
    should not need to.
    """
    ctx: dict[str, str] = {"request_id": request_id}
    if user_id:
        ctx["user_id"] = user_id
    if patient_id:
        ctx["patient_id"] = patient_id
    if session_id:
        ctx["session_id"] = session_id
    structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Clear structlog context-vars at the end of a request."""
    structlog.contextvars.clear_contextvars()