"""
FastAPI application factory.

Responsibilities
----------------
  - Create the FastAPI app with metadata and OpenAPI configuration.
  - Register all middleware in the correct order.
  - Register all route routers (REST v1 + WebSocket).
  - Register global exception handlers so all KinesioBaseError subclasses
    produce consistent JSON error responses.
  - Manage startup/shutdown lifespan events:
      startup:  DB pool, Redis pool, TimescaleDB pool, service instances.
      shutdown: graceful close of all pools and the Claude HTTP client.

Middleware execution order
--------------------------
FastAPI applies middleware in reverse registration order, so the last
add_middleware() call runs first on each request:

  Registration order:     Execution order (request in):
    CORSMiddleware     →    RequestIDMiddleware  (outermost)
    RateLimitMiddleware→    RateLimitMiddleware
    AuthMiddleware     →    AuthMiddleware
    RequestIDMiddleware→    CORSMiddleware        (innermost)

Service instantiation
---------------------
All service objects are created once here and stored on app.state.
Route handlers and dependencies pull them from app.state via deps.py.
This avoids re-creating clients (especially ClaudeClient with its HTTP
connection pool) on every request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import KinesioBaseError, RateLimitExceededError
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager managing the full application lifecycle.

    Everything before `yield` runs at startup; everything after runs at shutdown.
    FastAPI guarantees the shutdown block runs even if startup partially fails.
    """
    configure_logging()
    log.info("app_startup", version=settings.APP_VERSION, env=settings.ENVIRONMENT)

    # ── Startup ───────────────────────────────────────────────────────────────

    # Database pools
    from app.db.postgres import create_db_pool
    from app.db.redis import create_redis_pool
    from app.db.timescale import create_timescale_pool

    await create_db_pool()
    redis = await create_redis_pool()
    await create_timescale_pool()

    # AI client (single shared instance with connection pool)
    from app.ai.claude_client import ClaudeClient
    claude = ClaudeClient()

    # Services (stateless — safe to share across requests)
    from app.api.ws.connection_manager import ConnectionManager
    from app.services.exercise_planner import ExercisePlannerService
    from app.services.feedback_generator import FeedbackGeneratorService
    from app.services.notification import NotificationService
    from app.services.plan_adapter import PlanAdapterService
    from app.services.pose_analyzer import PoseAnalyzerService
    from app.services.recovery_forecaster import RecoveryForecasterService
    from app.services.red_flag_monitor import RedFlagMonitorService
    from app.services.session_manager import SessionManagerService
    from app.services.session_scorer import SessionScorerService
    from app.services.video_intake_analyzer import VideoIntakeAnalyzerService

    app.state.redis               = redis
    app.state.claude_client       = claude
    app.state.exercise_planner    = ExercisePlannerService(claude)
    app.state.plan_adapter        = PlanAdapterService(claude)
    app.state.pose_analyzer       = PoseAnalyzerService()
    app.state.feedback_generator  = FeedbackGeneratorService(claude, redis)
    app.state.session_manager     = SessionManagerService(redis)
    app.state.session_scorer      = SessionScorerService()
    app.state.recovery_forecaster = RecoveryForecasterService()
    app.state.red_flag_monitor    = RedFlagMonitorService(claude)
    app.state.video_intake_analyzer = VideoIntakeAnalyzerService()
    app.state.notification_service  = NotificationService()
    app.state.connection_manager  = ConnectionManager(redis)

    log.info("app_startup_complete", services_initialised=True)

    yield   # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────

    log.info("app_shutdown_started")

    await claude.close()

    from app.db.postgres import close_db_pool
    from app.db.redis import close_redis_pool
    from app.db.timescale import close_timescale_pool

    await close_db_pool()
    await close_redis_pool()
    await close_timescale_pool()

    log.info("app_shutdown_complete")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Kinesio-Track API",
        description=(
            "Remote physiotherapy agent — AI-powered exercise plans, "
            "real-time pose feedback, and recovery tracking."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        # Only expose docs in non-production environments
        docs_url="/docs"    if not settings.is_production else None,
        redoc_url="/redoc"  if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    _register_middleware(app)
    _register_routers(app)
    _register_exception_handlers(app)
    _register_health_check(app)

    return app


# ── Middleware ────────────────────────────────────────────────────────────────

def _register_middleware(app: FastAPI) -> None:
    """
    Register middleware in reverse execution order.
    Last registered = first to run on incoming requests.
    """
    from app.core.middleware import AuthMiddleware, RateLimitMiddleware, RequestIDMiddleware

    # CORS — innermost, applied after auth
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [
            "https://app.kinesiotrack.com",
            "https://www.kinesiotrack.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Rate limiting — after auth so we can rate-limit by user_id
    app.add_middleware(RateLimitMiddleware)

    # Auth — decodes JWT, attaches user to request.state
    app.add_middleware(AuthMiddleware)

    # Request ID — outermost, stamps every request before anything else
    app.add_middleware(RequestIDMiddleware)


# ── Routers ────────────────────────────────────────────────────────────────────

def _register_routers(app: FastAPI) -> None:
    from app.api.v1 import v1_router
    from app.api.ws import ws_router

    app.include_router(v1_router)
    app.include_router(ws_router)


# ── Exception handlers ────────────────────────────────────────────────────────

def _register_exception_handlers(app: FastAPI) -> None:
    """
    Map KinesioBaseError subclasses to consistent JSON error responses.

    Response format:
        {
          "error": {
            "code":       "snake_case_error_code",
            "message":    "Human-readable description",
            "detail":     {...} | null,
            "request_id": "uuid"
          }
        }
    """

    @app.exception_handler(KinesioBaseError)
    async def kinesio_error_handler(
        request: Request, exc: KinesioBaseError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        log.warning(
            "handled_exception",
            error_code=exc.error_code,
            message=exc.message,
            status=exc.http_status,
            request_id=request_id,
        )
        headers = {}
        if isinstance(exc, RateLimitExceededError):
            headers["Retry-After"] = str(exc.retry_after)

        return JSONResponse(
            status_code=exc.http_status,
            headers=headers,
            content={
                "error": {
                    "code":       exc.error_code,
                    "message":    exc.message,
                    "detail":     exc.detail,
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        log.error(
            "unhandled_exception",
            error=str(exc),
            exc_info=exc,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code":       "internal_error",
                    "message":    "An unexpected error occurred.",
                    "detail":     None,
                    "request_id": request_id,
                }
            },
        )


# ── Health check ──────────────────────────────────────────────────────────────

def _register_health_check(app: FastAPI) -> None:

    @app.get("/health", include_in_schema=False)
    async def health(request: Request) -> dict:
        """
        Liveness probe for load balancers and container orchestrators.
        Returns 200 if the process is alive.
        Performs a shallow DB ping to confirm connectivity.
        """
        db_ok = False
        redis_ok = False

        try:
            from sqlalchemy import text
            from app.db.postgres import get_engine
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            pass

        try:
            redis = request.app.state.redis
            await redis.ping()
            redis_ok = True
        except Exception:
            pass

        status = "ok" if (db_ok and redis_ok) else "degraded"
        code   = 200 if status == "ok" else 503

        return JSONResponse(
            status_code=code,
            content={
                "status":  status,
                "version": settings.APP_VERSION,
                "db":      "ok" if db_ok    else "error",
                "redis":   "ok" if redis_ok else "error",
            },
        )


# ── Entry point ───────────────────────────────────────────────────────────────

app = create_app()