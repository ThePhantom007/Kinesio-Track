"""
app/core/config.py

Pydantic BaseSettings — single source of truth for all environment variables.
Validated at startup; any missing required var raises a clear error before the
server accepts traffic.
"""

from __future__ import annotations

import secrets
from typing import Literal

from pydantic import AnyUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "kinesio-track"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    # Generated at startup if not supplied; override in prod with a stable secret.
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str
    # asyncpg pool settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    # TTL for cached Claude feedback messages (keyed by exercise_id + error_type)
    REDIS_FEEDBACK_CACHE_TTL: int = 86_400       # 24 h
    # TTL for per-session landmark_rules cached at session start
    REDIS_SESSION_RULES_TTL: int = 7_200         # 2 h
    # TTL for JWT refresh token revocation list entries
    REDIS_REVOCATION_TTL: int = 60 * 60 * 24 * 7  # 7 days

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Anthropic ─────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    # Max attempts including the first call
    ANTHROPIC_MAX_RETRIES: int = 3
    ANTHROPIC_TIMEOUT_SECONDS: float = 60.0
    # Hard cap on tokens for the short real-time feedback prompt
    ANTHROPIC_FEEDBACK_MAX_TOKENS: int = 80
    # Max tokens for full plan generation
    ANTHROPIC_PLAN_MAX_TOKENS: int = 4_096
    # Monthly budget (USD) — alerts are sent when this is exceeded
    MONTHLY_TOKEN_BUDGET_USD: float = 100.0

    # ── AWS / S3 ──────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    # Pre-signed URL lifetime (seconds)
    S3_PRESIGNED_EXPIRES: int = 900   # 15 min
    # Max video file size accepted (bytes)
    S3_MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024  # 500 MB

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    MEDIAPIPE_MODEL_PATH: str = "mediapipe/models/pose_landmarker.task"
    # 0 = lite, 1 = full, 2 = heavy
    MEDIAPIPE_MODEL_COMPLEXITY: int = 1
    # Minimum landmark visibility score to trust a keypoint
    MEDIAPIPE_MIN_VISIBILITY: float = 0.65

    # ── Pose Analyser ─────────────────────────────────────────────────────────
    # Degrees of deviation that produce a "warning" feedback event
    POSE_WARNING_THRESHOLD: float = 12.0
    # Degrees of deviation that produce an "error" and may trigger red-flag check
    POSE_ERROR_THRESHOLD: float = 22.0
    # Consecutive frames in violation before a feedback event fires
    POSE_VIOLATION_FRAME_COUNT: int = 3

    # ── Session / Analytics ───────────────────────────────────────────────────
    # Minimum completed sessions before recovery ETA is calculated
    MIN_SESSIONS_FOR_ETA: int = 3
    # Quality score (0–100) below which plan regression is considered
    QUALITY_REGRESSION_THRESHOLD: float = 45.0
    # Quality score above which phase progression is considered
    QUALITY_PROGRESSION_THRESHOLD: float = 78.0
    # Pain score (1–10) that triggers an automatic red-flag check
    PAIN_RED_FLAG_THRESHOLD: int = 8

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    # Requests per minute (sliding window via Redis)
    RATE_LIMIT_INTAKE: int = 5
    RATE_LIMIT_API_DEFAULT: int = 120
    RATE_LIMIT_WS_CONNECT: int = 20
    RATE_LIMIT_AUTH: int = 10

    # ── Notifications ─────────────────────────────────────────────────────────
    FCM_SERVER_KEY: str = ""
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@kinesiotrack.app"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not (v.startswith("postgresql") or v.startswith("sqlite")):
            raise ValueError("DATABASE_URL must be a PostgreSQL (or SQLite for tests) URL")
        # Ensure asyncpg driver is specified
        if "postgresql://" in v and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://")
        return v

    @model_validator(mode="after")
    def warn_insecure_defaults(self) -> "Settings":
        if self.is_production and self.DEBUG:
            raise ValueError("DEBUG must be False in production")
        return self

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def access_token_expire_seconds(self) -> int:
        return self.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    @property
    def refresh_token_expire_seconds(self) -> int:
        return self.REFRESH_TOKEN_EXPIRE_DAYS * 86_400


# Singleton — import this everywhere
settings = Settings()