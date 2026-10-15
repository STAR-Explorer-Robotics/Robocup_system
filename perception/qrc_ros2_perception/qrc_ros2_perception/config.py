"""Configuration loading for the ROS2 sensing bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from rescue_inspection.runtime import resolve_repo_root, resolve_runtime_output_root


DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_OVERLAY_TOPIC = "/qrc/sensing/overlay/image_raw"
DEFAULT_RESULT_TOPIC = "/qrc/sensing/result_json"
DEFAULT_LATEST_CONFIRMED_TOPIC = "/qrc/sensing/latest_confirmed_json"
DEFAULT_LATEST_CONFIRMED_OVERLAY_TOPIC = "/qrc/sensing/latest_confirmed_overlay/image_raw"
DEFAULT_STATUS_TOPIC = "/qrc/sensing/status"
DEFAULT_CAMERA_PASSTHROUGH_TOPIC = DEFAULT_IMAGE_TOPIC
VALID_INPUT_MODES = {"ros_image_topic", "pyrealsense2"}
VALID_PYREALSENSE_FORMATS = {"rgb8"}
VALID_CAMERA_PASSTHROUGH_ENCODINGS = {"bgr8"}


def detect_project_root() -> Path:
    """Locate the qrc_ros2_perception project root."""

    env_root = os.environ.get("QRC_ROS2_PERCEPTION_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    repo_project_root = resolve_repo_root() / "qrc_ros2_perception"
    if (repo_project_root / "package.xml").exists() and (repo_project_root / "qrc_ros2_perception").exists():
        return repo_project_root

    cwd = Path.cwd().resolve()
    if (cwd / "package.xml").exists() and (cwd / "qrc_ros2_perception").exists():
        return cwd

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = detect_project_root()


@dataclass(frozen=True)
class PerceptionBridgeConfig:
    """Runtime settings for qrc_sensing_node."""

    input_mode: str = "ros_image_topic"
    image_topic: str = DEFAULT_IMAGE_TOPIC
    camera_info_topic: str | None = None
    input_qos: str = "sensor_data"
    pyrealsense_width: int = 640
    pyrealsense_height: int = 480
    pyrealsense_fps: int = 15
    pyrealsense_serial: str | None = None
    pyrealsense_format: str = "rgb8"
    inspection_config: Path = Path("../inspection/configs/sensing_runtime_dl.yaml")
    publish_overlay: bool = True
    overlay_publish_fps: float = 10.0
    enable_local_overlay_window: bool = False
    local_overlay_window_fps: float = 15.0
    local_overlay_window_name: str = "QRC Overlay"
    local_overlay_wait_key_ms: int = 1
    publish_camera_passthrough: bool = False
    camera_passthrough_topic: str = DEFAULT_CAMERA_PASSTHROUGH_TOPIC
    camera_passthrough_fps: float = 10.0
    camera_passthrough_frame_id: str = "camera_color_optical_frame"
    camera_passthrough_encoding: str = "bgr8"
    publish_result_json: bool = True
    publish_latest_confirmed_json: bool = True
    max_processing_fps: float | None = None
    max_frames: int | None = None
    drop_frames_when_busy: bool = True
    overlay_topic: str = DEFAULT_OVERLAY_TOPIC
    result_topic: str = DEFAULT_RESULT_TOPIC
    latest_confirmed_topic: str = DEFAULT_LATEST_CONFIRMED_TOPIC
    latest_confirmed_overlay_topic: str = DEFAULT_LATEST_CONFIRMED_OVERLAY_TOPIC
    status_topic: str = DEFAULT_STATUS_TOPIC
    status_period_sec: float = 1.0
    save_outputs: bool = True
    output_root: Path = Path("outputs")
    save_overlay_video: bool = True
    save_confirmed_overlays: bool = True
    latest_result_write_period_sec: float = 0.0
    output_video_fps: float = 10.0
    log_level: str = "INFO"


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a qrc_ros2_perception-relative path."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _validate_config(config: PerceptionBridgeConfig) -> PerceptionBridgeConfig:
    input_mode = str(config.input_mode).strip()
    if input_mode not in VALID_INPUT_MODES:
        allowed = ", ".join(sorted(VALID_INPUT_MODES))
        raise ValueError(f"input_mode must be one of: {allowed}")

    pyrealsense_format = str(config.pyrealsense_format).strip().lower()
    if pyrealsense_format not in VALID_PYREALSENSE_FORMATS:
        allowed = ", ".join(sorted(VALID_PYREALSENSE_FORMATS))
        raise ValueError(f"pyrealsense_format must be one of: {allowed}")

    camera_passthrough_encoding = str(config.camera_passthrough_encoding).strip().lower()
    if camera_passthrough_encoding not in VALID_CAMERA_PASSTHROUGH_ENCODINGS:
        allowed = ", ".join(sorted(VALID_CAMERA_PASSTHROUGH_ENCODINGS))
        raise ValueError(f"camera_passthrough_encoding must be one of: {allowed}")

    return replace(
        config,
        input_mode=input_mode,
        pyrealsense_width=_positive_int(config.pyrealsense_width, field_name="pyrealsense_width"),
        pyrealsense_height=_positive_int(config.pyrealsense_height, field_name="pyrealsense_height"),
        pyrealsense_fps=_positive_int(config.pyrealsense_fps, field_name="pyrealsense_fps"),
        pyrealsense_serial=_optional_str(config.pyrealsense_serial),
        pyrealsense_format=pyrealsense_format,
        camera_passthrough_topic=str(config.camera_passthrough_topic).strip(),
        camera_passthrough_fps=float(config.camera_passthrough_fps),
        camera_passthrough_frame_id=str(config.camera_passthrough_frame_id).strip(),
        camera_passthrough_encoding=camera_passthrough_encoding,
    )


def _coerce_config(raw: dict[str, Any]) -> PerceptionBridgeConfig:
    defaults = PerceptionBridgeConfig()
    payload = raw.get("qrc_ros2_perception", raw)
    return _validate_config(
        PerceptionBridgeConfig(
            input_mode=str(payload.get("input_mode", defaults.input_mode)),
            image_topic=str(payload.get("image_topic", defaults.image_topic)),
            camera_info_topic=_optional_str(payload.get("camera_info_topic", defaults.camera_info_topic)),
            input_qos=str(payload.get("input_qos", defaults.input_qos)),
            pyrealsense_width=_positive_int(
                payload.get("pyrealsense_width", defaults.pyrealsense_width),
                field_name="pyrealsense_width",
            ),
            pyrealsense_height=_positive_int(
                payload.get("pyrealsense_height", defaults.pyrealsense_height),
                field_name="pyrealsense_height",
            ),
            pyrealsense_fps=_positive_int(
                payload.get("pyrealsense_fps", defaults.pyrealsense_fps),
                field_name="pyrealsense_fps",
            ),
            pyrealsense_serial=_optional_str(payload.get("pyrealsense_serial", defaults.pyrealsense_serial)),
            pyrealsense_format=str(payload.get("pyrealsense_format", defaults.pyrealsense_format)),
            inspection_config=Path(payload.get("inspection_config", defaults.inspection_config)),
            publish_overlay=bool(payload.get("publish_overlay", defaults.publish_overlay)),
            overlay_publish_fps=float(payload.get("overlay_publish_fps", defaults.overlay_publish_fps)),
            enable_local_overlay_window=bool(
                payload.get("enable_local_overlay_window", defaults.enable_local_overlay_window)
            ),
            local_overlay_window_fps=float(
                payload.get("local_overlay_window_fps", defaults.local_overlay_window_fps)
            ),
            local_overlay_window_name=str(
                payload.get("local_overlay_window_name", defaults.local_overlay_window_name)
            ),
            local_overlay_wait_key_ms=int(
                payload.get("local_overlay_wait_key_ms", defaults.local_overlay_wait_key_ms)
            ),
            publish_camera_passthrough=bool(
                payload.get("publish_camera_passthrough", defaults.publish_camera_passthrough)
            ),
            camera_passthrough_topic=str(
                payload.get("camera_passthrough_topic", defaults.camera_passthrough_topic)
            ),
            camera_passthrough_fps=float(
                payload.get("camera_passthrough_fps", defaults.camera_passthrough_fps)
            ),
            camera_passthrough_frame_id=str(
                payload.get("camera_passthrough_frame_id", defaults.camera_passthrough_frame_id)
            ),
            camera_passthrough_encoding=str(
                payload.get("camera_passthrough_encoding", defaults.camera_passthrough_encoding)
            ),
            publish_result_json=bool(payload.get("publish_result_json", defaults.publish_result_json)),
            publish_latest_confirmed_json=bool(
                payload.get("publish_latest_confirmed_json", defaults.publish_latest_confirmed_json)
            ),
            max_processing_fps=_optional_float(payload.get("max_processing_fps", defaults.max_processing_fps)),
            max_frames=_optional_int(payload.get("max_frames", defaults.max_frames)),
            drop_frames_when_busy=bool(payload.get("drop_frames_when_busy", defaults.drop_frames_when_busy)),
            overlay_topic=str(payload.get("overlay_topic", defaults.overlay_topic)),
            result_topic=str(payload.get("result_topic", defaults.result_topic)),
            latest_confirmed_topic=str(payload.get("latest_confirmed_topic", defaults.latest_confirmed_topic)),
            latest_confirmed_overlay_topic=str(
                payload.get("latest_confirmed_overlay_topic", defaults.latest_confirmed_overlay_topic)
            ),
            status_topic=str(payload.get("status_topic", defaults.status_topic)),
            status_period_sec=float(payload.get("status_period_sec", defaults.status_period_sec)),
            save_outputs=bool(payload.get("save_outputs", defaults.save_outputs)),
            output_root=Path(payload.get("output_root", defaults.output_root)),
            save_overlay_video=bool(payload.get("save_overlay_video", defaults.save_overlay_video)),
            save_confirmed_overlays=bool(payload.get("save_confirmed_overlays", defaults.save_confirmed_overlays)),
            latest_result_write_period_sec=float(
                payload.get("latest_result_write_period_sec", defaults.latest_result_write_period_sec)
            ),
            output_video_fps=float(payload.get("output_video_fps", defaults.output_video_fps)),
            log_level=str(payload.get("log_level", defaults.log_level)),
        )
    )


def load_perception_config(path: Path | str | None = None) -> PerceptionBridgeConfig:
    """Load bridge YAML config, resolving model/config paths relative to this project."""

    if path is None or str(path).strip() == "":
        config = PerceptionBridgeConfig()
    else:
        config_path = resolve_project_path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"qrc_ros2_perception config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = _coerce_config(yaml.safe_load(handle) or {})

    return _validate_config(
        replace(
            config,
            inspection_config=resolve_project_path(config.inspection_config),
            output_root=resolve_runtime_output_root(config.output_root, base=PROJECT_ROOT),
        )
    )


def apply_parameter_overrides(
    config: PerceptionBridgeConfig,
    *,
    image_topic: str | None = None,
    overlay_topic: str | None = None,
    result_topic: str | None = None,
    latest_confirmed_topic: str | None = None,
    latest_confirmed_overlay_topic: str | None = None,
    input_mode: str | None = None,
    pyrealsense_width: int | None = None,
    pyrealsense_height: int | None = None,
    pyrealsense_fps: int | None = None,
    pyrealsense_serial: str | None = None,
    pyrealsense_format: str | None = None,
    publish_camera_passthrough: bool | None = None,
    camera_passthrough_topic: str | None = None,
    camera_passthrough_fps: float | None = None,
    camera_passthrough_frame_id: str | None = None,
    camera_passthrough_encoding: str | None = None,
    inspection_config: str | None = None,
    output_root: str | None = None,
    use_sim_time: bool | None = None,
) -> PerceptionBridgeConfig:
    """Apply launch/ROS parameter overrides that should win over YAML."""

    del use_sim_time
    updates: dict[str, object] = {}
    if image_topic:
        updates["image_topic"] = image_topic
    if overlay_topic:
        updates["overlay_topic"] = overlay_topic
    if result_topic:
        updates["result_topic"] = result_topic
    if latest_confirmed_topic:
        updates["latest_confirmed_topic"] = latest_confirmed_topic
    if latest_confirmed_overlay_topic:
        updates["latest_confirmed_overlay_topic"] = latest_confirmed_overlay_topic
    if input_mode:
        updates["input_mode"] = input_mode
    if pyrealsense_width is not None:
        updates["pyrealsense_width"] = pyrealsense_width
    if pyrealsense_height is not None:
        updates["pyrealsense_height"] = pyrealsense_height
    if pyrealsense_fps is not None:
        updates["pyrealsense_fps"] = pyrealsense_fps
    if pyrealsense_serial is not None:
        updates["pyrealsense_serial"] = pyrealsense_serial
    if pyrealsense_format:
        updates["pyrealsense_format"] = pyrealsense_format
    if publish_camera_passthrough is not None:
        updates["publish_camera_passthrough"] = publish_camera_passthrough
    if camera_passthrough_topic:
        updates["camera_passthrough_topic"] = camera_passthrough_topic
    if camera_passthrough_fps is not None:
        updates["camera_passthrough_fps"] = camera_passthrough_fps
    if camera_passthrough_frame_id:
        updates["camera_passthrough_frame_id"] = camera_passthrough_frame_id
    if camera_passthrough_encoding:
        updates["camera_passthrough_encoding"] = camera_passthrough_encoding
    if inspection_config:
        updates["inspection_config"] = resolve_project_path(inspection_config)
    if output_root:
        updates["output_root"] = resolve_runtime_output_root(output_root, base=PROJECT_ROOT)
    return _validate_config(replace(config, **updates))
