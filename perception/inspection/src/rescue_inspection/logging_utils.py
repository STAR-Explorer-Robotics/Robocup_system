"""Logging setup helpers."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the package logger."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("rescue_inspection")

