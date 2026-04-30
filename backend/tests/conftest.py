"""
Shared pytest fixtures used across unit, integration, and e2e test suites.

Fixture scopes
--------------
  session-scoped:  DB engine, async event loop — created once per test run.
  function-scoped: DB session (with per-test rollback), HTTP client,
                   Redis mock — isolated per test.

Database strategy
-----------------
Integration tests use a real PostgreSQL instance (spun up by docker-compose.test.yml
or a locally running Postgres with the DATABASE_URL env var pointing to a test DB).
Each test gets its own transaction that is rolled back on teardown, so tests
are fully isolated without truncating tables between runs.

External service mocking
------------------------
  - Claude API:  mocked via respx (intercepts httpx calls at the transport layer).
  - Redis:       replaced with fakeredis.aioredis for speed and isolation.
  - S3:          mocked via moto (in-memory S3).
  - Celery:      configured with ALWAYS_EAGER=True so tasks run inline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import create_access_token
from app.main import create_app
from app.models import Base


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop shared across the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Database ──────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = settings.DATABASE_URL.replace(
    "kinesiotrack", "kinesiotrack_test"
)

@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create the test DB schema once per test session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Per-test transactional session that rolls back on teardown.
    Tests can flush and query freely; nothing persists to the DB.
    """
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ── Redis mock ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def mock_redis():
    """In-memory Redis replacement using fakeredis."""
    try:
        import fakeredis.aioredis as fakeredis
        redis = await fakeredis.create_redis_pool()
    except ImportError:
        # Fallback: plain AsyncMock if fakeredis not installed
        redis = AsyncMock()
        redis.get    = AsyncMock(return_value=None)
        redis.set    = AsyncMock(return_value=True)
        redis.setex  = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=1)
        redis.exists = AsyncMock(return_value=0)
        redis.lpush  = AsyncMock(return_value=1)
        redis.lrange = AsyncMock(return_value=[])
        redis.incr   = AsyncMock(return_value=1)
        redis.hset   = AsyncMock(return_value=1)
        redis.hget   = AsyncMock(return_value=None)
        redis.hgetall= AsyncMock(return_value={})
        redis.expire = AsyncMock(return_value=True)
        redis.ping   = AsyncMock(return_value=True)
        redis.publish= AsyncMock(return_value=0)
        redis.scan   = AsyncMock(return_value=(0, []))
        redis.ttl    = AsyncMock(return_value=3600)
        redis.pipeline = MagicMock(return_value=redis)
        redis.execute  = AsyncMock(return_value=[0, 0, 1, True])
    yield redis


# ── Claude mock ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_claude_client():
    """
    Mock ClaudeClient that returns fixture responses without hitting the API.
    Import and use the fixture responses from tests/fixtures/mock_claude_responses.py.
    """
    from tests.fixtures.mock_claude_responses import (
        VALID_PLAN_RESPONSE,
        VALID_PATCH_RESPONSE,
        VALID_RED_FLAG_RESPONSE,
        VALID_FEEDBACK_RESPONSE,
    )

    client = AsyncMock()
    client.generate_initial_plan = AsyncMock(return_value=VALID_PLAN_RESPONSE)
    client.adapt_plan            = AsyncMock(return_value=VALID_PATCH_RESPONSE)
    client.escalate_red_flag     = AsyncMock(return_value=VALID_RED_FLAG_RESPONSE)
    client.generate_feedback     = AsyncMock(return_value=VALID_FEEDBACK_RESPONSE)
    client.close                 = AsyncMock()
    return client


# ── HTTP client ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db_session, mock_redis, mock_claude_client) -> AsyncGenerator[AsyncClient, None]:
    """
    Async test HTTP client wired to the FastAPI app.
    Overrides DB, Redis, and Claude with test doubles.
    """
    app = create_app()

    # Wire test doubles onto app.state
    from app.services.exercise_planner    import ExercisePlannerService
    from app.services.plan_adapter        import PlanAdapterService
    from app.services.pose_analyzer       import PoseAnalyzerService
    from app.services.feedback_generator  import FeedbackGeneratorService
    from app.services.session_manager     import SessionManagerService
    from app.services.session_scorer      import SessionScorerService
    from app.services.recovery_forecaster import RecoveryForecasterService
    from app.services.red_flag_monitor    import RedFlagMonitorService
    from app.services.video_intake_analyzer import VideoIntakeAnalyzerService
    from app.services.notification        import NotificationService
    from app.api.ws.connection_manager    import ConnectionManager

    app.state.redis                = mock_redis
    app.state.claude_client        = mock_claude_client
    app.state.exercise_planner     = ExercisePlannerService(mock_claude_client)
    app.state.plan_adapter         = PlanAdapterService(mock_claude_client)
    app.state.pose_analyzer        = PoseAnalyzerService()
    app.state.feedback_generator   = FeedbackGeneratorService(mock_claude_client, mock_redis)
    app.state.session_manager      = SessionManagerService(mock_redis)
    app.state.session_scorer       = SessionScorerService()
    app.state.recovery_forecaster  = RecoveryForecasterService()
    app.state.red_flag_monitor     = RedFlagMonitorService(mock_claude_client)
    app.state.video_intake_analyzer = VideoIntakeAnalyzerService()
    app.state.notification_service = NotificationService()
    app.state.connection_manager   = ConnectionManager(mock_redis)

    # Override DB dependency
    from app.db.postgres import get_db
    from app.db.redis import get_redis

    app.dependency_overrides[get_db]    = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ── Auth tokens ───────────────────────────────────────────────────────────────

@pytest.fixture
def patient_token(sample_patient_user) -> str:
    """Valid JWT access token for the sample patient."""
    return create_access_token(
        str(sample_patient_user.id),
        "patient",
    )


@pytest.fixture
def clinician_token(sample_clinician_user) -> str:
    """Valid JWT access token for the sample clinician."""
    return create_access_token(
        str(sample_clinician_user.id),
        "clinician",
    )


@pytest.fixture
def auth_headers(patient_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {patient_token}"}


@pytest.fixture
def clinician_headers(clinician_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {clinician_token}"}


# ── Sample ORM fixtures ────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def sample_patient_user(db_session: AsyncSession):
    from app.core.security import hash_password
    from app.models.user import User, UserRole

    user = User(
        email="testpatient@kinesiotest.local",
        hashed_password=hash_password("TestPass1!"),
        full_name="Test Patient",
        role=UserRole.PATIENT,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def sample_clinician_user(db_session: AsyncSession):
    from app.core.security import hash_password
    from app.models.user import User, UserRole

    user = User(
        email="testclinician@kinesiotest.local",
        hashed_password=hash_password("TestPass1!"),
        full_name="Test Clinician",
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def sample_patient(db_session: AsyncSession, sample_patient_user):
    from app.models.patient import PatientProfile

    profile = PatientProfile(
        user_id=sample_patient_user.id,
        baseline_rom={
            "left_ankle": {"angle_deg": 18.5, "frame_index": 100},
        },
    )
    db_session.add(profile)
    await db_session.flush()
    return profile


@pytest_asyncio.fixture
async def sample_injury(db_session: AsyncSession, sample_patient):
    from app.models.injury import BodyPart, Injury, InjuryStatus

    injury = Injury(
        patient_id=sample_patient.id,
        description="Left ankle sprain from running.",
        body_part=BodyPart.ANKLE,
        pain_score=6,
        status=InjuryStatus.ACTIVE,
    )
    db_session.add(injury)
    await db_session.flush()
    return injury


@pytest_asyncio.fixture
async def sample_plan(db_session: AsyncSession, sample_patient, sample_injury):
    from app.models.exercise import Exercise
    from app.models.phase import PlanPhase
    from app.models.plan import ExercisePlan, PlanStatus

    plan = ExercisePlan(
        patient_id=sample_patient.id,
        injury_id=sample_injury.id,
        title="Ankle Sprain Recovery",
        version=1,
        status=PlanStatus.ACTIVE,
        current_phase=1,
        recovery_target_days=42,
        ai_generated=True,
        contraindications=[],
        escalation_criteria=[
            {"trigger": "pain_score >= 8", "action": "stop", "reason": "Pain spike"},
        ],
    )
    db_session.add(plan)
    await db_session.flush()

    phase = PlanPhase(
        plan_id=plan.id,
        phase_number=1,
        name="Phase 1 – Acute Recovery",
        goal="Restore baseline ankle mobility",
        duration_days=14,
    )
    db_session.add(phase)
    await db_session.flush()

    exercise = Exercise(
        phase_id=phase.id,
        slug="seated-ankle-circles",
        name="Seated Ankle Circles",
        order_index=0,
        sets=3,
        reps=10,
        hold_seconds=0,
        rest_seconds=30,
        target_joints=["left_ankle", "right_ankle"],
        landmark_rules={
            "left_ankle": {
                "min_angle": 10.0,
                "max_angle": 35.0,
                "axis": "sagittal",
                "priority": "primary",
            },
        },
        red_flags=[],
        patient_instructions="Rotate your ankle slowly in full circles.",
        difficulty="beginner",
    )
    db_session.add(exercise)
    await db_session.flush()

    sample_patient.active_plan_id = plan.id
    db_session.add(sample_patient)
    await db_session.flush()

    return plan


@pytest_asyncio.fixture
async def sample_session(db_session: AsyncSession, sample_patient, sample_plan):
    from sqlalchemy import select
    from app.models.exercise import Exercise
    from app.models.phase import PlanPhase
    from app.models.session import ExerciseSession, SessionStatus
    from datetime import timedelta

    result = await db_session.execute(
        select(Exercise)
        .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
        .where(PlanPhase.plan_id == sample_plan.id)
        .limit(1)
    )
    exercise = result.scalar_one()

    now = datetime.now(timezone.utc)
    session = ExerciseSession(
        patient_id=sample_patient.id,
        plan_id=sample_plan.id,
        exercise_id=exercise.id,
        status=SessionStatus.COMPLETED,
        started_at=now - timedelta(minutes=30),
        ended_at=now,
        post_session_pain=4,
        completion_pct=0.9,
        avg_quality_score=72.5,
        total_reps_completed=27,
        total_sets_completed=3,
        peak_rom_degrees=28.4,
    )
    db_session.add(session)
    await db_session.flush()
    return session


# ── Landmark fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def good_form_landmarks():
    """33 landmarks representing correct ankle circle form."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "sample_landmarks.json"
    if fixture_path.exists():
        data = json.loads(fixture_path.read_text())
        return data.get("ankle_circle_correct", _synthetic_landmarks())
    return _synthetic_landmarks()


@pytest.fixture
def knee_valgus_landmarks():
    """33 landmarks with left knee collapsed inward (valgus)."""
    lm = _synthetic_landmarks()
    # Push left knee medially by shifting x inward
    lm[25]["x"] = lm[23]["x"] + 0.05   # knee closer to midline than hip
    return lm


def _synthetic_landmarks() -> list[dict]:
    """Generate 33 neutral-pose landmarks at anatomically plausible positions."""
    # Simplified upright standing pose normalised to [0,1]
    positions = {
        0:  (0.50, 0.05),   # nose
        7:  (0.46, 0.08),   # left ear
        8:  (0.54, 0.08),   # right ear
        11: (0.44, 0.28),   # left shoulder
        12: (0.56, 0.28),   # right shoulder
        13: (0.42, 0.42),   # left elbow
        14: (0.58, 0.42),   # right elbow
        15: (0.40, 0.55),   # left wrist
        16: (0.60, 0.55),   # right wrist
        19: (0.39, 0.60),   # left index
        20: (0.61, 0.60),   # right index
        23: (0.46, 0.55),   # left hip
        24: (0.54, 0.55),   # right hip
        25: (0.46, 0.72),   # left knee
        26: (0.54, 0.72),   # right knee
        27: (0.46, 0.88),   # left ankle
        28: (0.54, 0.88),   # right ankle
        31: (0.46, 0.96),   # left foot index
        32: (0.54, 0.96),   # right foot index
    }

    lm = []
    for i in range(33):
        x, y = positions.get(i, (0.50, 0.50))
        lm.append({
            "id":         i,
            "x":          x,
            "y":          y,
            "z":          -0.02,
            "visibility": 0.95,
        })
    return lm