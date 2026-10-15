"""Configuration loading for the latest confirmed result panel."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LATEST_CONFIRMED_TOPIC = "/qrc/sensing/latest_confirmed_json"
DEFAULT_LATEST_CONFIRMED_OVERLAY_TOPIC = "/qrc/sensing/latest_confirmed_overlay/image_raw"
DEFAULT_STATUS_TOPIC = "/qrc/sensing/status"
DEFAULT_PANEL_IMAGE_TOPIC = "/qrc/sensing/result_panel/image_raw"


@dataclass(frozen=True)
class ResultPanelConfig:
    """Runtime settings for qrc_result_panel."""

    latest_confirmed_topic: str = DEFAULT_LATEST_CONFIRMED_TOPIC
    latest_confirmed_overlay_topic: str = DEFAULT_LATEST_CONFIRMED_OVERLAY_TOPIC
    status_topic: str = DEFAULT_STATUS_TOPIC
    panel_image_topic: str = DEFAULT_PANEL_IMAGE_TOPIC
    publish_panel_image: bool = True
    show_window: bool = True
    window_name: str = "Latest Confirmed Result Window"
    refresh_fps: float = 2.0
    stale_threshold_sec: float = 10.0
    panel_width: int = 900
    panel_height: int = 500


def detect_project_root() -> Path:
    """Locate the qrc_ros2_perception project root without importing inspection."""

    env_root = os.environ.get("QRC_ROS2_PERCEPTION_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "package.xml").exists() and (cwd / "qrc_ros2_perception").exists():
        return cwd
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = detect_project_root()


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a qrc_ros2_perception-relative path."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_result_panel_config(path: Path | str | None = None) -> ResultPanelConfig:
    """Load result panel YAML config."""

    if path is None or str(path).strip() == "":
        return ResultPanelConfig()
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"result panel config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _coerce_config(raw)


def apply_result_panel_overrides(
    config: ResultPanelConfig,
    *,
    latest_confirmed_topic: str | None = None,
    latest_confirmed_overlay_topic: str | None = None,
    status_topic: str | None = None,
    panel_image_topic: str | None = None,
    publish_panel_image: bool | None = None,
    show_window: bool | None = None,
    window_name: str | None = None,
    refresh_fps: float | None = None,
    stale_threshold_sec: float | None = None,
    panel_width: int | None = None,
    panel_height: int | None = None,
) -> ResultPanelConfig:
    """Apply ROS parameter overrides on top of YAML config."""

    updates: dict[str, object] = {}
    if latest_confirmed_topic:
        updates["latest_confirmed_topic"] = latest_confirmed_topic
    if latest_confirmed_overlay_topic:
        updates["latest_confirmed_overlay_topic"] = latest_confirmed_overlay_topic
    if status_topic:
        updates["status_topic"] = status_topic
    if panel_image_topic:
        updates["panel_image_topic"] = panel_image_topic
    if publish_panel_image is not None:
        updates["publish_panel_image"] = bool(publish_panel_image)
    if show_window is not None:
        updates["show_window"] = bool(show_window)
    if window_name:
        updates["window_name"] = window_name
    if refresh_fps is not None:
        updates["refresh_fps"] = float(refresh_fps)
    if stale_threshold_sec is not None:
        updates["stale_threshold_sec"] = float(stale_threshold_sec)
    if panel_width is not None:
        updates["panel_width"] = int(panel_width)
    if panel_height is not None:
        updates["panel_height"] = int(panel_height)
    return replace(config, **updates)


def _coerce_config(raw: dict[str, Any]) -> ResultPanelConfig:
    defaults = ResultPanelConfig()
    payload = raw.get("result_panel", raw)
    return ResultPanelConfig(
        latest_confirmed_topic=str(payload.get("latest_confirmed_topic", defaults.latest_confirmed_topic)),
        latest_confirmed_overlay_topic=str(
            payload.get("latest_confirmed_overlay_topic", defaults.latest_confirmed_overlay_topic)
        ),
        status_topic=str(payload.get("status_topic", defaults.status_topic)),
        panel_image_topic=str(payload.get("panel_image_topic", defaults.panel_image_topic)),
        publish_panel_image=bool(payload.get("publish_panel_image", defaults.publish_panel_image)),
        show_window=bool(payload.get("show_window", defaults.show_window)),
        window_name=str(payload.get("window_name", defaults.window_name)),
        refresh_fps=float(payload.get("refresh_fps", defaults.refresh_fps)),
        stale_threshold_sec=float(payload.get("stale_threshold_sec", defaults.stale_threshold_sec)),
        panel_width=int(payload.get("panel_width", defaults.panel_width)),
        panel_height=int(payload.get("panel_height", defaults.panel_height)),
    )
