"""Optional pyrealsense2 color frame source for the ROS2 sensing bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PyRealSense2Frame:
    """A BGR frame plus lightweight source metadata."""

    image_bgr: np.ndarray
    metadata: dict[str, object]


class PyRealSense2ColorSource:
    """Read color frames from a RealSense device with lazy pyrealsense2 import."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        serial: str | None = None,
        color_format: str = "rgb8",
        timeout_ms: int = 1000,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial.strip() if serial else None
        self.color_format = color_format.strip().lower()
        self.timeout_ms = int(timeout_ms)
        self._rs: Any = None
        self._pipeline: Any = None
        self._actual_profile: dict[str, object] | None = None

    @property
    def requested_profile(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.color_format,
            "serial": self.serial,
        }

    @property
    def actual_profile(self) -> dict[str, object] | None:
        return self._actual_profile

    def open(self) -> None:
        if self.color_format != "rgb8":
            raise ValueError("pyrealsense_format must be one of: rgb8")
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "pyrealsense2 is not installed; rebuild the qrc_ros2_perception development "
                "container or install pyrealsense2 in the active ROS2 environment."
            ) from exc

        self.close()
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            if _looks_like_device_busy(exc):
                raise RuntimeError(
                    "RealSense device is busy. Stop realsense2_camera, realsense-viewer, "
                    "other pyrealsense2 programs, and older QRC containers before starting "
                    "input_mode=pyrealsense2."
                ) from exc
            raise

        self._rs = rs
        self._pipeline = pipeline
        self._actual_profile = _actual_profile_info(profile, rs, fallback=self.requested_profile)

    def read_bgr_frame(self) -> PyRealSense2Frame | None:
        if self._pipeline is None:
            raise RuntimeError("pyrealsense2 source is not open")

        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
        except RuntimeError as exc:
            if _looks_like_timeout(exc):
                return None
            raise

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        rgb = np.asanyarray(color_frame.get_data())
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        metadata = {
            "source_time_sec": _source_time_sec(color_frame),
            "actual_profile": self._actual_profile,
        }
        return PyRealSense2Frame(image_bgr=image_bgr, metadata=metadata)

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._rs = None
        if pipeline is not None:
            pipeline.stop()


def _looks_like_timeout(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "timeout" in message or "frame didn't arrive" in message or "frame did not arrive" in message


def _looks_like_device_busy(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "resource busy" in message or "errno=16" in message or "device or resource busy" in message


def _source_time_sec(color_frame: Any) -> float | None:
    try:
        return float(color_frame.get_timestamp()) / 1000.0
    except Exception:
        return None


def _actual_profile_info(profile: Any, rs: Any, *, fallback: dict[str, object]) -> dict[str, object]:
    info = dict(fallback)
    try:
        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        info.update(
            {
                "width": int(stream_profile.width()),
                "height": int(stream_profile.height()),
                "fps": int(stream_profile.fps()),
                "format": str(stream_profile.format()),
            }
        )
    except Exception:
        pass

    try:
        device = profile.get_device()
        info["serial"] = str(device.get_info(rs.camera_info.serial_number))
    except Exception:
        pass

    return info
