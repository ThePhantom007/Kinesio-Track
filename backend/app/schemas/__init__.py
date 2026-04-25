"""
Re-exports every public schema so route handlers and services can import from
``app.schemas`` directly rather than from individual sub-modules.
"""

from app.schemas.auth import (  # noqa: F401
    AuthResponse,
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.base import AppBaseModel, AppResponseModel, PaginatedResponse  # noqa: F401
from app.schemas.intake import InjuryIntakeRequest, InjuryIntakeResponse  # noqa: F401
from app.schemas.media import (  # noqa: F401
    MediaFileResponse,
    ProcessConfirmResponse,
    ProcessingNotifyRequest,
    UploadUrlRequest,
    UploadUrlResponse,
)
from app.schemas.patient import PatientResponse, PatientSummary, PatientUpdateRequest  # noqa: F401
from app.schemas.plan import (  # noqa: F401
    ExerciseAIOutput,
    ExercisePlanAIOutput,
    ExercisePlanResponse,
    ExercisePlanSummary,
    ExerciseResponse,
    ExerciseSummary,
    JointRule,
    PlanPatchRequest,
    PlanPhaseAIOutput,
    PlanPhaseResponse,
    RedFlagRule,
)
from app.schemas.progress import (  # noqa: F401
    JointROMSeries,
    ProgressMilestone,
    ProgressQueryParams,
    ProgressResponse,
    QualityDataPoint,
    RecoveryForecast,
    ROMDataPoint,
)
from app.schemas.session import (  # noqa: F401
    SessionEndRequest,
    SessionHistoryResponse,
    SessionListItem,
    SessionMetrics,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryResponse,
)
from app.schemas.websocket import (  # noqa: F401
    ErrorMessage,
    ExerciseDoneMessage,
    FeedbackMessage,
    InboundMessage,
    Landmark,
    LandmarkFrame,
    MilestoneMessage,
    OutboundMessage,
    OverlayPoint,
    PingMessage,
    PongMessage,
    RedFlagMessage,
    RepCompleteMessage,
    SessionSummaryMessage,
    WSCloseCode,
)