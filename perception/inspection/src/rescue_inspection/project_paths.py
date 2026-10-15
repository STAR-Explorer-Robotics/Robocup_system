"""Compatibility wrappers for inspection-root-relative path helpers."""

from __future__ import annotations

from pathlib import Path

from rescue_inspection.runtime.paths import (
    relative_to_inspection_root,
    resolve_inspection_relative_path,
    resolve_inspection_root,
)


def inspection_root() -> Path:
    """Return the `inspection/` project root."""

    return resolve_inspection_root()


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a path relative to the project root when needed."""

    return resolve_inspection_relative_path(path)


def relative_to_root(path: Path | str) -> str:
    """Return a stable project-root-relative string when possible."""

    return relative_to_inspection_root(path)
