"""
Declarative base and shared mixins inherited by every ORM model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Root declarative base — all models inherit from this."""
    pass


class TimestampMixin:
    """
    Adds created_at and updated_at columns to any model.
    updated_at is refreshed automatically by the DB on every UPDATE via
    onupdate=func.now().
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """
    Adds a UUID primary key column.  UUIDs are generated in Python so the
    value is available before the INSERT round-trip completes.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class BaseModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Convenience base that combines UUID PK + timestamps.
    Most models should inherit from this rather than Base directly.
    """

    __abstract__ = True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id}>"