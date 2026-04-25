"""
Re-exports every service class for convenient imports throughout the app.
Services are instantiated in app/main.py lifespan and stored on app.state,
or injected via FastAPI Depends() in app/api/deps.py.
"""

from app.services.exercise_planner import ExercisePlannerService  # noqa: F401
from app.services.feedback_generator import FeedbackGeneratorService  # noqa: F401
from app.services.notification import NotificationService  # noqa: F401
from app.services.plan_adapter import PlanAdapterService  # noqa: F401
from app.services.pose_analyzer import (  # noqa: F401
    FrameAnalysisResult,
    JointViolation,
    PoseAnalyzerService,
)
from app.services.recovery_forecaster import RecoveryForecasterService  # noqa: F401
from app.services.red_flag_monitor import RedFlagMonitorService  # noqa: F401
from app.services.session_manager import SessionManagerService  # noqa: F401
from app.services.session_scorer import SessionMetrics, SessionScorerService  # noqa: F401
from app.services.video_intake_analyzer import VideoIntakeAnalyzerService  # noqa: F401