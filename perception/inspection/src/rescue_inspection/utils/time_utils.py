"""Time-related helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def iso_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    """Return a filesystem-friendly run identifier."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"
