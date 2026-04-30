"""
Pre-flight environment validation.

Checks that every required environment variable is set AND that each
external service is actually reachable.  Run this in the Docker entrypoint
before starting Uvicorn or Celery so problems surface immediately with a
clear error message rather than as cryptic runtime failures.

Usage:
    Python scripts/check_env.py              # exit 0 on success, 1 on failure
    Python scripts/check_env.py --no-ping   # validate vars only, skip pings

Exit codes:
    0  All checks passed.
    1  One or more checks failed (details printed to stderr).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root so app imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Required variables ────────────────────────────────────────────────────────

REQUIRED_VARS: list[tuple[str, str]] = [
    ("DATABASE_URL",          "PostgreSQL connection string"),
    ("REDIS_URL",             "Redis connection URL"),
    ("ANTHROPIC_API_KEY",     "Anthropic Claude API key"),
    ("AWS_ACCESS_KEY_ID",     "AWS / MinIO access key"),
    ("AWS_SECRET_ACCESS_KEY", "AWS / MinIO secret key"),
    ("S3_BUCKET_NAME",        "S3 bucket for video storage"),
    ("JWT_SECRET",            "JWT signing secret (min 32 chars)"),
]

OPTIONAL_VARS: list[tuple[str, str]] = [
    ("FCM_SERVER_KEY",   "Firebase Cloud Messaging key (push notifications)"),
    ("SMTP_USER",        "SMTP username (email alerts)"),
    ("SMTP_PASSWORD",    "SMTP password (email alerts)"),
]


# ── Checks ────────────────────────────────────────────────────────────────────

def check_vars() -> list[str]:
    """Return list of error strings for missing or invalid env vars."""
    errors: list[str] = []

    for var, desc in REQUIRED_VARS:
        val = os.getenv(var)
        if not val:
            errors.append(f"  MISSING  {var}  ({desc})")
            continue

        # Extra validation for specific vars
        if var == "JWT_SECRET" and len(val) < 32:
            errors.append(f"  WEAK     {var}  (must be ≥ 32 characters, got {len(val)})")

        if var == "DATABASE_URL" and not (
            val.startswith("postgresql") or val.startswith("sqlite")
        ):
            errors.append(f"  INVALID  {var}  (must start with 'postgresql')")

        if var == "ANTHROPIC_API_KEY" and not val.startswith("sk-ant-"):
            errors.append(f"  INVALID  {var}  (expected 'sk-ant-...' format)")

    return errors


async def ping_postgres() -> str | None:
    """Return None on success, error string on failure."""
    try:
        import asyncpg
        url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn=url, timeout=5)
        await conn.execute("SELECT 1")
        await conn.close()
        return None
    except Exception as exc:
        return f"PostgreSQL unreachable: {exc}"


async def ping_redis() -> str | None:
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=5,
        )
        await client.ping()
        await client.aclose()
        return None
    except Exception as exc:
        return f"Redis unreachable: {exc}"


async def ping_s3() -> str | None:
    try:
        import asyncio
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        def _head():
            client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            )
            bucket = os.getenv("S3_BUCKET_NAME", "")
            client.head_bucket(Bucket=bucket)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _head)
        return None
    except Exception as exc:
        return f"S3 unreachable or bucket missing: {exc}"


async def ping_anthropic() -> str | None:
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        # Cheapest possible call to verify the key is valid
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        await client.close()
        return None
    except anthropic.AuthenticationError:
        return "Anthropic API key is invalid or expired"
    except Exception as exc:
        return f"Anthropic API unreachable: {exc}"


# ── Runner ────────────────────────────────────────────────────────────────────

async def main(no_ping: bool) -> int:
    print("Kinesio-Track — pre-flight environment check\n")

    # Step 1: required vars
    print("[ 1/2 ] Checking environment variables…")
    var_errors = check_vars()
    if var_errors:
        print("  FAILED:")
        for e in var_errors:
            print(e)
    else:
        print(f"  OK  — {len(REQUIRED_VARS)} required vars present")

    # Optional vars (warnings only)
    missing_optional = [v for v, _ in OPTIONAL_VARS if not os.getenv(v)]
    if missing_optional:
        print(f"  WARN — optional vars not set: {', '.join(missing_optional)}")

    if var_errors:
        print("\n✗ Variable check failed. Set the missing vars and retry.\n")
        return 1

    if no_ping:
        print("\n✓ Variable check passed (--no-ping, skipping connectivity).\n")
        return 0

    # Step 2: connectivity
    print("\n[ 2/2 ] Pinging external services…")
    ping_results: dict[str, str | None] = {}

    tasks = {
        "PostgreSQL":  ping_postgres(),
        "Redis":       ping_redis(),
        "S3":          ping_s3(),
        "Anthropic":   ping_anthropic(),
    }

    for name, coro in tasks.items():
        result = await coro
        ping_results[name] = result
        status = "OK" if result is None else "FAIL"
        detail = f" — {result}" if result else ""
        print(f"  {status:4s}  {name}{detail}")

    failed = [name for name, err in ping_results.items() if err]

    if failed:
        print(f"\n✗ Connectivity check failed for: {', '.join(failed)}\n")
        return 1

    print("\n✓ All checks passed — environment is ready.\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Kinesio-Track environment.")
    parser.add_argument(
        "--no-ping",
        action="store_true",
        help="Skip external service connectivity checks.",
    )
    args = parser.parse_args()

    # Load .env if present (for local runs outside Docker)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    exit_code = asyncio.run(main(no_ping=args.no_ping))
    sys.exit(exit_code)