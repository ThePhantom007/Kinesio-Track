"""
Three ASGI middlewares registered in app/main.py:

  1. RequestIDMiddleware   — stamps every request/response with X-Request-ID
                             and binds it to structlog context-vars so every
                             log line emitted during that request carries it.

  2. RateLimitMiddleware   — sliding-window rate limiter backed by Redis.
                             Per-route limits are configured in config.py.
                             Returns 429 with a Retry-After header on breach.

  3. AuthMiddleware        — validates the Bearer JWT on protected routes,
                             attaches the decoded user payload to
                             request.state.user, and rejects revoked tokens
                             by checking a Redis revocation list.

Usage (app/main.py):
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)
    # Note: middlewares are applied in reverse-registration order,
    # so RequestID runs first, then RateLimit, then Auth.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.exceptions import AuthenticationError, RateLimitExceededError, TokenExpiredError
from app.core.logging import bind_request_context, clear_request_context, get_logger
from app.core.security import decode_access_token, revocation_key

log = get_logger(__name__)

# ── Route configuration ────────────────────────────────────────────────────────

# Public routes that bypass JWT auth entirely.
_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/api/v1/auth/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# Per-route rate limits: (prefix, requests_per_minute).
# First matching prefix wins; fall back to RATE_LIMIT_API_DEFAULT.
_RATE_LIMIT_RULES: list[tuple[str, int]] = [
    ("/api/v1/auth/",    settings.RATE_LIMIT_AUTH),
    ("/api/v1/intake",   settings.RATE_LIMIT_INTAKE),
    ("/ws/",             settings.RATE_LIMIT_WS_CONNECT),
]

# Routes that are exempt from rate limiting altogether (health / metrics).
_RATE_LIMIT_EXEMPT: tuple[str, ...] = ("/health", "/metrics")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_public(path: str) -> bool:
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def _is_rate_limit_exempt(path: str) -> bool:
    return any(path.startswith(p) for p in _RATE_LIMIT_EXEMPT)


def _get_rate_limit(path: str) -> int:
    for prefix, limit in _RATE_LIMIT_RULES:
        if path.startswith(prefix):
            return limit
    return settings.RATE_LIMIT_API_DEFAULT


def _error_response(status: int, code: str, message: str, **extra) -> JSONResponse:
    body = {"error": {"code": code, "message": message, **extra}}
    return JSONResponse(status_code=status, content=body)


# ═════════════════════════════════════════════════════════════════════════════
# 1. RequestIDMiddleware
# ═════════════════════════════════════════════════════════════════════════════

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Stamps every inbound request with a UUID ``X-Request-ID`` header and
    reflects it on the response.  If the client already supplies the header
    (e.g. an internal service hop), the existing value is preserved.

    Also binds the request_id to structlog context-vars for the duration of
    the request so that all log entries carry it automatically.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Expose on request.state so route handlers / services can access it.
        request.state.request_id = request_id

        # Bind to structlog context for this async task.
        bind_request_context(request_id=request_id)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=getattr(response, "status_code", 0),
                duration_ms=round(elapsed_ms, 2),
            )
            clear_request_context()

        response.headers["X-Request-ID"] = request_id
        return response


# ═════════════════════════════════════════════════════════════════════════════
# 2. RateLimitMiddleware
# ═════════════════════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter backed by Redis.

    Algorithm
    ---------
    For each (identifier, route-bucket) pair we maintain a sorted set in Redis
    where each member is a unique request UUID and the score is the request
    timestamp in milliseconds.

    On every request:
      1. Remove all members with score < (now - window_ms).
      2. Count remaining members.
      3. If count >= limit → reject with 429.
      4. Otherwise add the new request to the set with ZADD.
      5. Set the key TTL to window_ms so Redis auto-cleans idle keys.

    Identifier priority: authenticated user_id > IP address.
    This prevents a single IP from poisoning shared IP-NAT environments once
    users are logged in, while still protecting the login endpoint by IP.
    """

    WINDOW_SECONDS: int = 60

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        if _is_rate_limit_exempt(path):
            return await call_next(request)

        limit = _get_rate_limit(path)
        identifier = self._get_identifier(request)
        redis = request.app.state.redis

        allowed, retry_after = await self._check_limit(redis, identifier, path, limit)

        if not allowed:
            log.warning(
                "rate_limit_exceeded",
                identifier=identifier,
                path=path,
                limit=limit,
                retry_after=retry_after,
            )
            return _error_response(
                429,
                "rate_limit_exceeded",
                f"Too many requests. Try again in {retry_after} seconds.",
                retry_after=retry_after,
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        return response

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_identifier(request: Request) -> str:
        """
        Return the most specific identifier available.
        Prefers authenticated user_id; falls back to client IP.
        """
        user = getattr(request.state, "user", None)
        if user and user.get("sub"):
            return f"user:{user['sub']}"
        forwarded_for = request.headers.get("X-Forwarded-For")
        ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
        return f"ip:{ip}"

    async def _check_limit(
        self,
        redis,
        identifier: str,
        path: str,
        limit: int,
    ) -> tuple[bool, int]:
        """
        Execute the sliding-window check.

        Returns:
            (allowed, retry_after_seconds)
        """
        now_ms = int(time.time() * 1000)
        window_ms = self.WINDOW_SECONDS * 1000
        window_start_ms = now_ms - window_ms

        # Bucket by first two path segments to group sub-routes.
        segments = [s for s in path.split("/") if s]
        bucket = "/".join(segments[:2])
        key = f"rl:{identifier}:{bucket}"

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start_ms)
        pipe.zcard(key)
        pipe.zadd(key, {str(uuid.uuid4()): now_ms})
        pipe.pexpire(key, window_ms)
        results = await pipe.execute()

        current_count = results[1]  # before adding this request
        if current_count >= limit:
            # Estimate how many ms until the oldest request falls out of window.
            oldest = await redis.zrange(key, 0, 0, withscores=True)
            retry_after = 60  # default
            if oldest:
                _, oldest_score = oldest[0]
                retry_after = max(1, int((oldest_score + window_ms - now_ms) / 1000))
            return False, retry_after

        return True, 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. AuthMiddleware
# ═════════════════════════════════════════════════════════════════════════════

class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT authentication middleware.

    Behaviour
    ---------
    • Public routes (defined in ``_PUBLIC_PREFIXES``) pass through without any
      token requirement.
    • All other routes require a valid ``Authorization: Bearer <token>`` header.
    • Decoded payload is attached to ``request.state.user`` so route handlers
      and ``get_current_user()`` dependency can read it without re-decoding.
    • Revocation check: the token's ``jti`` is looked up in Redis.  If found,
      the token was invalidated by a logout or a security rotation and is
      rejected even if the signature and expiry are valid.
    • User identity (user_id, patient_id) is bound to the structlog context so
      all subsequent log lines carry it.

    WebSocket note
    --------------
    WebSocket upgrade requests arrive via GET; the JWT is expected either in
    the ``Authorization`` header (preferred) or as a ``?token=`` query param
    (Android fallback when the WS library can't set headers).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if _is_public(request.url.path):
            request.state.user = None
            return await call_next(request)

        token = self._extract_token(request)
        if not token:
            return _error_response(
                401,
                "authentication_error",
                "Missing Authorization header or token query parameter.",
            )

        redis = request.app.state.redis

        try:
            payload = decode_access_token(token)
        except TokenExpiredError:
            return _error_response(401, "token_expired", "Access token has expired.")
        except AuthenticationError as exc:
            return _error_response(401, "authentication_error", str(exc))

        # Revocation check
        jti = payload.get("jti")
        if jti:
            is_revoked = await redis.exists(revocation_key(jti))
            if is_revoked:
                log.warning("revoked_token_used", jti=jti, user_id=payload.get("sub"))
                return _error_response(
                    401,
                    "token_revoked",
                    "This token has been revoked. Please log in again.",
                )

        request.state.user = payload

        # Enrich structlog context with identity for downstream log lines.
        bind_request_context(
            request_id=getattr(request.state, "request_id", ""),
            user_id=payload.get("sub"),
        )

        log.debug(
            "auth_ok",
            user_id=payload.get("sub"),
            role=payload.get("role"),
            path=request.url.path,
        )

        return await call_next(request)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        """
        Extract the raw JWT string from the request.

        Priority:
          1. ``Authorization: Bearer <token>`` header.
          2. ``?token=<token>`` query parameter (WebSocket fallback).
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer "):]

        # WebSocket query-param fallback
        if request.url.path.startswith("/ws/"):
            token_param = request.query_params.get("token")
            if token_param:
                return token_param

        return None