"""
Imports every ORM model so that:
  1. Alembic autogenerate sees all tables when it inspects the metadata.
  2. SQLAlchemy relationship() resolution works correctly at mapper
     configuration time (all classes must be in memory before the first
     mapper configure() call).

Import order follows the foreign-key dependency graph — referenced tables
before referencing tables — to avoid mapper configuration errors on startup.
"""

from app.models.base import Base, BaseModel  # noqa: F401

# ── No FKs ────────────────────────────────────────────────────────────────────
from app.models.user import User, UserRole  # noqa: F401

# ── Depend on User ────────────────────────────────────────────────────────────
from app.models.patient import ActivityLevel, PatientProfile  # noqa: F401
from app.models.clinician import ClinicianPatient, ClinicianProfile  # noqa: F401

# ── Depend on PatientProfile ──────────────────────────────────────────────────
from app.models.injury import BodyPart, Injury, InjuryStatus  # noqa: F401
from app.models.token_usage import AICallType, TokenUsage  # noqa: F401

# ── Depend on Injury ──────────────────────────────────────────────────────────
from app.models.plan import ExercisePlan, PlanStatus  # noqa: F401

# ── Depend on ExercisePlan ────────────────────────────────────────────────────
from app.models.phase import PlanPhase  # noqa: F401

# ── Depend on PlanPhase ───────────────────────────────────────────────────────
from app.models.exercise import Exercise  # noqa: F401

# ── Depend on ExercisePlan + Exercise ─────────────────────────────────────────
from app.models.session import ExerciseSession, SessionStatus  # noqa: F401

# ── Depend on ExerciseSession ─────────────────────────────────────────────────
from app.models.media import MediaFile, MediaType, ProcessingStatus  # noqa: F401
from app.models.feedback_event import FeedbackEvent, FeedbackSeverity  # noqa: F401
from app.models.red_flag import RedFlagEvent, RedFlagSeverity, RedFlagTrigger  # noqa: F401

__all__ = [
    # Base
    "Base",
    "BaseModel",
    # Users
    "User",
    "UserRole",
    # Profiles
    "PatientProfile",
    "ActivityLevel",
    "ClinicianProfile",
    "ClinicianPatient",
    # Clinical
    "Injury",
    "InjuryStatus",
    "BodyPart",
    # Plans
    "ExercisePlan",
    "PlanStatus",
    "PlanPhase",
    "Exercise",
    # Sessions
    "ExerciseSession",
    "SessionStatus",
    # Media & events
    "MediaFile",
    "MediaType",
    "ProcessingStatus",
    "FeedbackEvent",
    "FeedbackSeverity",
    "RedFlagEvent",
    "RedFlagSeverity",
    "RedFlagTrigger",
    # AI cost tracking
    "TokenUsage",
    "AICallType",
]