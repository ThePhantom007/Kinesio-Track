"""
Shared Pydantic v2 base configuration and reusable field types used across
all schema modules.

Rules applied everywhere:
  - from_attributes=True so ORM models can be passed directly to model_validate().
  - populate_by_name=True so both alias and Python name are accepted on input.
  - str_strip_whitespace=True to silently trim leading/trailing whitespace from
    all string fields before validation.
  - Responses use datetime with timezone; naive datetimes are rejected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_serializer
from pydantic.functional_validators import BeforeValidator


# ── Base config ────────────────────────────────────────────────────────────────

class AppBaseModel(BaseModel):
    """Root Pydantic model — all schemas inherit from this."""

    model_config = ConfigDict(
        from_attributes=True,        # allow ORM → schema conversion
        populate_by_name=True,       # accept field name OR alias
        str_strip_whitespace=True,   # trim strings automatically
        use_enum_values=True,        # serialise enums to their .value
        arbitrary_types_allowed=True,
    )


class AppResponseModel(AppBaseModel):
    """
    Base for all *response* schemas.
    Enforces timezone-aware datetimes in serialised output so the frontend
    always receives ISO-8601 strings with a UTC offset.
    """

    @field_validator("*", mode="before")
    @classmethod
    def _ensure_tz_aware(cls, v: Any) -> Any:
        if isinstance(v, datetime) and v.tzinfo is None:
            # Assume UTC for any naive datetime coming from the DB.
            return v.replace(tzinfo=timezone.utc)
        return v


# ── Reusable annotated types ───────────────────────────────────────────────────

def _coerce_uuid(v: Any) -> UUID:
    """Accept both UUID objects and UUID strings."""
    if isinstance(v, UUID):
        return v
    try:
        return UUID(str(v))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid UUID: {v!r}") from exc


UUIDField = Annotated[UUID, BeforeValidator(_coerce_uuid)]


# ── Pagination ────────────────────────────────────────────────────────────────

class PaginatedResponse(AppResponseModel):
    """Generic wrapper for cursor-paginated list endpoints."""

    items: list[Any]
    total: int
    next_cursor: str | None = None
    has_more: bool = False