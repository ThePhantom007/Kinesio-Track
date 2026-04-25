"""
Async Redis client with a shared connection pool.

Used for:
  - WebSocket session state (landmark_rules, feedback buffer, rep counters)
  - Claude feedback message cache (exercise_slug + error_type → message)
  - JWT revocation list (jti → expiry timestamp)
  - Rate limiter sliding-window sorted sets
  - Celery broker and result backend (configured separately in celery_app.py)

Connection lifecycle
--------------------
create_redis_pool() and close_redis_pool() are called from the FastAPI
lifespan context manager in app/main.py.  The client is stored on app.state
so it can be injected via get_redis() in route handlers and WebSocket endpoints.

Pub/Sub
-------
The connection_manager in app/api/ws/connection_manager.py uses Redis Pub/Sub
to broadcast messages across multiple Uvicorn worker processes.  Each worker
subscribes to a channel keyed by session_id; the WebSocket handler publishes
to the channel and all subscribed workers deliver to their connected clients.
"""

from __future__ import annotations

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Module-level singleton — populated by create_redis_pool().
_redis_client: Redis | None = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def create_redis_pool() -> Redis:
    """
    Create the async Redis connection pool.
    Returns the client so it can be stored on app.state.
    """
    global _redis_client

    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,       # always return str, not bytes
        max_connections=50,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
    )

    # Verify connectivity at startup
    await _redis_client.ping()
    log.info("redis_pool_created", url=_sanitised_redis_url(settings.REDIS_URL))
    return _redis_client


async def close_redis_pool() -> None:
    """Close the Redis connection pool. Called during FastAPI app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        log.info("redis_pool_closed")


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_redis() -> Redis:
    """
    FastAPI dependency that returns the shared Redis client.

    Usage::

        @router.get("/example")
        async def my_route(redis: Redis = Depends(get_redis)):
            await redis.set("key", "value", ex=60)
    """
    if _redis_client is None:
        raise RuntimeError("Redis pool has not been initialised. Call create_redis_pool() first.")
    return _redis_client


def get_redis_client() -> Redis:
    """
    Synchronous accessor for use in Celery tasks where await is not available
    at the point of import but an event loop is running.
    """
    if _redis_client is None:
        raise RuntimeError("Redis pool has not been initialised.")
    return _redis_client


# ── Typed helper wrappers ─────────────────────────────────────────────────────
# These thin wrappers provide a consistent interface and centralise
# error handling so callers don't scatter try/except across the codebase.

async def cache_get(key: str) -> str | None:
    """Return the value for *key*, or None if it does not exist."""
    if _redis_client is None:
        return None
    try:
        return await _redis_client.get(key)
    except Exception as exc:
        log.warning("redis_get_error", key=key, error=str(exc))
        return None


async def cache_set(key: str, value: str, ttl: int | None = None) -> bool:
    """
    Set *key* → *value*.

    Args:
        key:   Redis key string.
        value: String value (all Redis values are strings with decode_responses=True).
        ttl:   Optional expiry in seconds.  If None, the key persists indefinitely.

    Returns:
        True on success, False on error.
    """
    if _redis_client is None:
        return False
    try:
        if ttl is not None:
            await _redis_client.setex(key, ttl, value)
        else:
            await _redis_client.set(key, value)
        return True
    except Exception as exc:
        log.warning("redis_set_error", key=key, error=str(exc))
        return False


async def cache_delete(*keys: str) -> int:
    """
    Delete one or more keys.

    Returns:
        Number of keys actually deleted.
    """
    if _redis_client is None or not keys:
        return 0
    try:
        return await _redis_client.delete(*keys)
    except Exception as exc:
        log.warning("redis_delete_error", keys=keys, error=str(exc))
        return 0


async def cache_exists(key: str) -> bool:
    """Return True if *key* exists in Redis."""
    if _redis_client is None:
        return False
    try:
        return bool(await _redis_client.exists(key))
    except Exception as exc:
        log.warning("redis_exists_error", key=key, error=str(exc))
        return False


# ── Pub/Sub ───────────────────────────────────────────────────────────────────

async def publish(channel: str, message: str) -> int:
    """
    Publish *message* to *channel*.

    Returns:
        Number of subscribers that received the message.
    """
    if _redis_client is None:
        return 0
    try:
        return await _redis_client.publish(channel, message)
    except Exception as exc:
        log.warning("redis_publish_error", channel=channel, error=str(exc))
        return 0


async def get_pubsub():
    """
    Return a new PubSub instance for subscribing to channels.
    The caller is responsible for calling .close() when done.

    Usage::

        pubsub = await get_pubsub()
        await pubsub.subscribe("session:abc123")
        async for message in pubsub.listen():
            if message["type"] == "message":
                handle(message["data"])
    """
    if _redis_client is None:
        raise RuntimeError("Redis pool has not been initialised.")
    return _redis_client.pubsub()


# ── JWT revocation ────────────────────────────────────────────────────────────

async def revoke_token(jti: str) -> None:
    """
    Add a token JTI to the revocation list.
    The key is set with the refresh token TTL so it auto-expires.
    """
    from app.core.security import revocation_key
    await cache_set(
        revocation_key(jti),
        "1",
        ttl=settings.REDIS_REVOCATION_TTL,
    )
    log.debug("token_revoked", jti=jti)


async def is_token_revoked(jti: str) -> bool:
    """Return True if the JTI is in the revocation list."""
    from app.core.security import revocation_key
    return await cache_exists(revocation_key(jti))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitised_redis_url(url: str) -> str:
    """Strip the password from a Redis URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url