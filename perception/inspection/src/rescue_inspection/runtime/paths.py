"""Runtime-friendly path resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_repo_root() -> Path:
    """Return the QRC repository root."""

    env_root = _env_path("QRC_ROOT")
    if env_root is not None:
        return env_root
    return Path(__file__).resolve().parents[4]


def resolve_inspection_root() -> Path:
    """Return the inspection project root."""

    env_root = _env_path("QRC_INSPECTION_ROOT")
    if env_root is not None:
        return env_root
    candidate = resolve_repo_root() / "inspection"
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[3]


def resolve_path(path: Path | str, *, base: Path | None = None) -> Path:
    """Resolve a path against a base directory without applying runtime overrides."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = base if base is not None else resolve_inspection_root()
    return (root / candidate).resolve()


def resolve_inspection_relative_path(path: Path | str) -> Path:
    """Resolve a path relative to the inspection project root."""

    return resolve_path(path, base=resolve_inspection_root())


def resolve_runtime_config_path(path: Path | str) -> Path:
    """Resolve a runtime config path with optional QRC_CONFIG_ROOT override."""

    return _resolve_runtime_rooted_path(
        path,
        env_var="QRC_CONFIG_ROOT",
        legacy_base=resolve_inspection_root(),
        stripped_prefix="configs",
    )


def resolve_runtime_model_path(path: Path | str) -> Path:
    """Resolve a runtime model path with optional QRC_RUNTIME_MODEL_ROOT override."""

    return _resolve_runtime_rooted_path(
        path,
        env_var="QRC_RUNTIME_MODEL_ROOT",
        legacy_base=resolve_inspection_root(),
        stripped_prefix="models",
    )


def resolve_runtime_output_root(path: Path | str | None = None, *, base: Path | None = None) -> Path:
    """Resolve a runtime output root with optional QRC_OUTPUT_ROOT override."""

    env_root = _env_path("QRC_OUTPUT_ROOT")
    if path is None:
        if env_root is not None:
            return env_root
        return resolve_path("outputs", base=base or resolve_inspection_root())

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if env_root is not None:
        return (env_root / _strip_prefix(candidate, "outputs")).resolve()
    return resolve_path(candidate, base=base or resolve_inspection_root())


def relative_to_inspection_root(path: Path | str) -> str:
    """Return a stable inspection-root-relative string when possible."""

    candidate = resolve_inspection_relative_path(path)
    try:
        return str(candidate.relative_to(resolve_inspection_root()))
    except ValueError:
        return str(candidate)


def _resolve_runtime_rooted_path(
    path: Path | str,
    *,
    env_var: str,
    legacy_base: Path,
    stripped_prefix: str,
) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    env_root = _env_path(env_var)
    if env_root is not None:
        return (env_root / _strip_prefix(candidate, stripped_prefix)).resolve()
    return resolve_path(candidate, base=legacy_base)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    candidate = Path(value).expanduser().resolve()
    
    # Safeguard against host-only absolute paths inheriting into Docker containers
    if os.path.exists("/.dockerenv") and candidate.is_absolute() and not candidate.exists():
        path_str = str(candidate)
        # Ignore absolute paths that are outside the container's standard workspace (/workspace) 
        # and test directory (/tmp), resolving fallback to the default container workspace path.
        if not (path_str.startswith("/workspace/") or path_str.startswith("/tmp/")):
            return None
            
    return candidate


def _strip_prefix(path: Path, prefix: str) -> Path:
    parts = path.parts
    if parts and parts[0] == prefix:
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return path
