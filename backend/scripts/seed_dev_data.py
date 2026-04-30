"""
Seeds the database with realistic development fixtures for fast local setup.

Creates:
  - 1 clinician user + profile
  - 2 patient users + profiles (one with a full session history)
  - 1 injury per patient
  - 1 active exercise plan per patient (3 phases, 4 exercises each)
  - 10 completed sessions with metrics for patient 1
  - Sample TimescaleDB metric rows for the progress dashboard
  - 2 feedback events per session
  - 1 red-flag event

Usage:
  Python scripts/seed_dev_data.py
  Python scripts/seed_dev_data.py --reset    # drops and re-seeds

All passwords are 'KinesioTest1!' (safe only for local dev).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path so app imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.core.security import hash_password
from app.db.postgres import create_db_pool, get_db_context
from app.db.timescale import create_timescale_pool, write_metric_batch
from app.models import (
    BodyPart,
    ClinicianProfile,
    Exercise,
    ExercisePlan,
    ExerciseSession,
    FeedbackEvent,
    FeedbackSeverity,
    Injury,
    InjuryStatus,
    PatientProfile,
    PlanPhase,
    PlanStatus,
    RedFlagEvent,
    RedFlagSeverity,
    RedFlagTrigger,
    SessionStatus,
    User,
    UserRole,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEV_PASSWORD = "KinesioTest1!"
JOINTS = ["left_ankle", "right_ankle", "left_knee", "right_knee"]

SAMPLE_LANDMARK_RULES = {
    "left_ankle":  {"min_angle": 10.0, "max_angle": 35.0, "axis": "sagittal", "priority": "primary"},
    "right_ankle": {"min_angle": 10.0, "max_angle": 35.0, "axis": "sagittal", "priority": "bilateral"},
}

SAMPLE_EXERCISES = [
    {
        "slug": "seated-ankle-circles",
        "name": "Seated Ankle Circles",
        "sets": 3, "reps": 10, "hold_seconds": 0, "rest_seconds": 30,
        "target_joints": ["left_ankle", "right_ankle"],
        "difficulty": "beginner",
        "patient_instructions": (
            "Sit in a chair with your feet flat on the floor. "
            "Lift one foot slightly and slowly rotate your ankle in a full circle, "
            "10 times clockwise then 10 times counter-clockwise."
        ),
    },
    {
        "slug": "calf-raises",
        "name": "Standing Calf Raises",
        "sets": 3, "reps": 15, "hold_seconds": 2, "rest_seconds": 45,
        "target_joints": ["left_ankle", "right_ankle", "left_knee", "right_knee"],
        "difficulty": "beginner",
        "patient_instructions": (
            "Stand near a wall for balance. Rise up onto your toes slowly, "
            "hold for 2 seconds, then lower back down with control."
        ),
    },
    {
        "slug": "resistance-band-eversion",
        "name": "Resistance Band Eversion",
        "sets": 3, "reps": 12, "hold_seconds": 1, "rest_seconds": 30,
        "target_joints": ["left_ankle"],
        "difficulty": "intermediate",
        "patient_instructions": (
            "Sit with your leg extended. Loop a resistance band around your foot. "
            "Slowly turn your foot outward against the band, hold, then return."
        ),
    },
    {
        "slug": "single-leg-balance",
        "name": "Single-Leg Balance",
        "sets": 3, "reps": 1, "hold_seconds": 30, "rest_seconds": 60,
        "target_joints": ["left_ankle", "left_knee", "left_hip"],
        "difficulty": "intermediate",
        "patient_instructions": (
            "Stand on your affected leg. Keep a slight bend in your knee. "
            "Hold for 30 seconds, focusing on keeping your ankle stable."
        ),
    },
]


# ── Main seeder ───────────────────────────────────────────────────────────────

async def seed(reset: bool = False) -> None:
    await create_db_pool()
    await create_timescale_pool()

    async with get_db_context() as db:
        if reset:
            print("⚠  Resetting seed data…")
            await _reset(db)

        print("→ Creating clinician…")
        clinician_user, clinician = await _create_clinician(db)

        print("→ Creating patient 1 (full history)…")
        patient1_user, patient1 = await _create_patient(
            db,
            email="patient1@kinesiodev.local",
            name="Priya Sharma",
            clinician=clinician,
        )

        print("→ Creating patient 2 (new, no sessions)…")
        patient2_user, patient2 = await _create_patient(
            db,
            email="patient2@kinesiodev.local",
            name="Arjun Mehta",
            clinician=clinician,
        )

        print("→ Creating injuries and plans…")
        injury1 = await _create_injury(db, patient1, BodyPart.ANKLE)
        injury2 = await _create_injury(db, patient2, BodyPart.KNEE)

        plan1 = await _create_plan(db, patient1, injury1)
        _     = await _create_plan(db, patient2, injury2)

        patient1.active_plan_id = plan1.id
        db.add(patient1)

        print("→ Creating 10 sessions for patient 1…")
        sessions = await _create_sessions(db, patient1, plan1)

        print("→ Writing TimescaleDB metrics…")
        await _write_metrics(sessions, plan1)

        print("→ Creating feedback events…")
        await _create_feedback_events(db, sessions)

        print("→ Creating red-flag event…")
        await _create_red_flag(db, patient1, sessions[-1], clinician)

    print("\n✓ Seed complete.")
    print(f"\n  Clinician  email: clinician@kinesiodev.local  password: {DEV_PASSWORD}")
    print(f"  Patient 1  email: patient1@kinesiodev.local   password: {DEV_PASSWORD}")
    print(f"  Patient 2  email: patient2@kinesiodev.local   password: {DEV_PASSWORD}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _reset(db) -> None:
    for email in [
        "clinician@kinesiodev.local",
        "patient1@kinesiodev.local",
        "patient2@kinesiodev.local",
    ]:
        result = await db.execute(
            text("DELETE FROM users WHERE email = :e"), {"e": email}
        )
    await db.flush()


async def _create_clinician(db):
    user = User(
        email="clinician@kinesiodev.local",
        hashed_password=hash_password(DEV_PASSWORD),
        full_name="Dr. Ananya Krishnan",
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    profile = ClinicianProfile(
        user_id=user.id,
        license_number="PT-MH-2024-0042",
        specialty="Sports Physiotherapy",
        institution="Kinesio Wellness Centre",
        email_alerts_enabled=True,
    )
    db.add(profile)
    await db.flush()
    return user, profile


async def _create_patient(db, email: str, name: str, clinician: ClinicianProfile):
    user = User(
        email=email,
        hashed_password=hash_password(DEV_PASSWORD),
        full_name=name,
        role=UserRole.PATIENT,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    profile = PatientProfile(
        user_id=user.id,
        date_of_birth=date(1992, 6, 15),
        region="Maharashtra, India",
        assigned_clinician_id=clinician.id,
        baseline_rom={
            "left_ankle":  {"angle_deg": 18.5, "frame_index": 142},
            "right_ankle": {"angle_deg": 31.2, "frame_index": 156},
            "left_knee":   {"angle_deg": 95.0, "frame_index": 200},
        },
        mobility_notes=(
            "Baseline ROM at intake:\n"
            "  Left Ankle: 18.5° (restricted)\n"
            "  Right Ankle: 31.2° (moderate restriction)\n"
            "  Left Knee: 95.0° (full range)\n"
            "Primary affected area: ankle. Reported pain: 6/10."
        ),
    )
    db.add(profile)
    await db.flush()
    return user, profile


async def _create_injury(db, patient: PatientProfile, body_part: BodyPart) -> Injury:
    descriptions = {
        BodyPart.ANKLE: (
            "Sprained my left ankle playing cricket 3 weeks ago. "
            "It twisted inward when I landed awkwardly. Significant swelling "
            "for the first week, now mostly resolved. Still feels unstable "
            "when walking on uneven ground and aches after standing for long."
        ),
        BodyPart.KNEE: (
            "Experiencing pain around the front of my right knee for the past "
            "month. Gets worse when climbing stairs or sitting for long periods. "
            "No acute injury — pain came on gradually. Occasional clicking sound."
        ),
    }
    injury = Injury(
        patient_id=patient.id,
        description=descriptions[body_part],
        body_part=body_part,
        pain_score=6,
        status=InjuryStatus.ACTIVE,
        mobility_notes="Baseline mobility measured from intake video.",
    )
    db.add(injury)
    await db.flush()
    return injury


async def _create_plan(db, patient: PatientProfile, injury: Injury) -> ExercisePlan:
    plan = ExercisePlan(
        patient_id=patient.id,
        injury_id=injury.id,
        title=f"{injury.body_part.value.replace('_', ' ').title()} Rehabilitation Programme",
        version=1,
        status=PlanStatus.ACTIVE,
        current_phase=1,
        recovery_target_days=42,
        ai_generated=True,
        contraindications=["Avoid full weight-bearing jumps", "No deep squats below 90°"],
        escalation_criteria=[
            {"trigger": "pain_score >= 8", "action": "stop", "reason": "Acute pain spike"},
            {"trigger": "bilateral_asymmetry > 25", "action": "warn", "reason": "Compensation pattern"},
        ],
    )
    db.add(plan)
    await db.flush()

    phases_data = [
        ("Phase 1 – Acute Recovery",   "Reduce pain and restore baseline mobility",    14),
        ("Phase 2 – Strength Building", "Rebuild ankle strength and proprioception",    14),
        ("Phase 3 – Return to Function","Restore full functional mobility and stability", 14),
    ]

    for phase_num, (name, goal, duration) in enumerate(phases_data, 1):
        phase = PlanPhase(
            plan_id=plan.id,
            phase_number=phase_num,
            name=name,
            goal=goal,
            duration_days=duration,
            progression_criteria=f"avg_quality_score >= 78 over 3 consecutive sessions",
        )
        db.add(phase)
        await db.flush()

        for order_idx, ex_data in enumerate(SAMPLE_EXERCISES):
            exercise = Exercise(
                phase_id=phase.id,
                slug=ex_data["slug"],
                name=ex_data["name"],
                order_index=order_idx,
                sets=ex_data["sets"],
                reps=ex_data["reps"],
                hold_seconds=ex_data["hold_seconds"],
                rest_seconds=ex_data["rest_seconds"],
                target_joints=ex_data["target_joints"],
                landmark_rules=SAMPLE_LANDMARK_RULES,
                red_flags=[
                    {"condition": "left_ankle.angle < 5", "action": "stop", "reason": "Hyperflexion risk"},
                ],
                patient_instructions=ex_data["patient_instructions"],
                difficulty=ex_data["difficulty"],
            )
            db.add(exercise)

    await db.flush()
    return plan


async def _create_sessions(
    db, patient: PatientProfile, plan: ExercisePlan
) -> list[ExerciseSession]:
    """Create 10 completed sessions spread over the last 20 days."""
    from sqlalchemy import select

    # Load first exercise of phase 1
    result = await db.execute(
        select(Exercise)
        .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
        .where(PlanPhase.plan_id == plan.id, PlanPhase.phase_number == 1)
        .order_by(Exercise.order_index)
        .limit(1)
    )
    exercise = result.scalar_one()

    sessions = []
    for i in range(10):
        days_ago = 20 - (i * 2)
        started  = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=random.randint(8, 18))
        ended    = started + timedelta(minutes=random.randint(20, 45))

        # Simulate improving quality over time
        quality = min(40.0 + (i * 5.5) + random.uniform(-3, 3), 95.0)
        pain    = max(7 - i, 1) + random.randint(-1, 1)
        pain    = max(1, min(10, pain))

        session = ExerciseSession(
            patient_id=patient.id,
            plan_id=plan.id,
            exercise_id=exercise.id,
            status=SessionStatus.COMPLETED,
            started_at=started,
            ended_at=ended,
            post_session_pain=pain,
            completion_pct=random.uniform(0.75, 1.0),
            avg_quality_score=round(quality, 1),
            total_reps_completed=random.randint(24, 36),
            total_sets_completed=3,
            peak_rom_degrees=round(18.5 + (i * 1.4), 1),
            plan_adapted=(i == 4),
            summary_text=(
                f"Session {i+1} complete. Form score {quality:.0f}%. "
                "Keep focusing on controlled ankle rotation."
            ),
        )
        db.add(session)
        sessions.append(session)

    await db.flush()
    return sessions


async def _write_metrics(sessions: list[ExerciseSession], plan: ExercisePlan) -> None:
    """Write per-session TimescaleDB metric rows."""
    rows = []
    for i, session in enumerate(sessions):
        base_angle = 18.5 + (i * 1.4)
        for joint in ["left_ankle", "right_ankle"]:
            for rep in range(10):
                rows.append({
                    "session_id":   str(session.id),
                    "exercise_id":  str(session.exercise_id),
                    "joint":        joint,
                    "angle_deg":    round(base_angle + random.uniform(-2, 2), 2),
                    "quality_score": round(session.avg_quality_score + random.uniform(-5, 5), 1),
                })
    if rows:
        await write_metric_batch(rows)


async def _create_feedback_events(db, sessions: list[ExerciseSession]) -> None:
    for session in sessions:
        for j in range(2):
            event = FeedbackEvent(
                session_id=session.id,
                exercise_id=session.exercise_id,
                occurred_at=session.started_at + timedelta(minutes=j * 5 + 2),
                severity=FeedbackSeverity.WARNING,
                error_type="ankle_insufficient_range",
                affected_joint="left_ankle",
                actual_angle=round(random.uniform(8, 14), 1),
                expected_min_angle=10.0,
                expected_max_angle=35.0,
                deviation_degrees=round(random.uniform(-5, -1), 1),
                form_score_at_event=round(session.avg_quality_score - 10, 1),
                message="Push your ankle a little further to reach the full range.",
                from_cache=True,
            )
            db.add(event)
    await db.flush()


async def _create_red_flag(
    db,
    patient: PatientProfile,
    session: ExerciseSession,
    clinician: ClinicianProfile,
) -> None:
    event = RedFlagEvent(
        patient_id=patient.id,
        session_id=session.id,
        trigger_type=RedFlagTrigger.PAIN_SPIKE,
        trigger_context={"pain_score": 8, "previous_avg_pain": 3.2, "increase": 4.8},
        severity=RedFlagSeverity.STOP,
        immediate_action=(
            "Please stop the exercise and rest. Your physiotherapist has been notified."
        ),
        clinician_note=(
            "Patient reported sudden pain spike to 8/10 during single-leg balance. "
            "Previous average was 3.2/10. Recommend reassessment before next session."
        ),
        session_recommendation="rest_and_reassess",
        claude_raw_response={"severity": "stop", "session_recommendation": "rest_and_reassess"},
        notification_method="email",
        clinician_notified_at=session.ended_at,
    )
    db.add(event)
    await db.flush()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Kinesio-Track dev database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing seed data before inserting.",
    )
    args = parser.parse_args()
    asyncio.run(seed(reset=args.reset))