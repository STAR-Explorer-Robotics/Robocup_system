"""ROS Image and OpenCV BGR conversion helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


SUPPORTED_INPUT_ENCODINGS = {"bgr8", "rgb8", "bgra8", "rgba8", "mono8"}


def image_msg_to_bgr(message: Any) -> np.ndarray:
    """Convert a ROS-like Image message to OpenCV BGR/HWC/uint8."""

    encoding = str(message.encoding).lower()
    if encoding not in SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"Unsupported ROS image encoding '{message.encoding}'. "
            f"Supported encodings: {sorted(SUPPORTED_INPUT_ENCODINGS)}"
        )

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got height={height} width={width}")

    channels = _channels_for_encoding(encoding)
    expected_visible_bytes = width * channels
    if step < expected_visible_bytes:
        raise ValueError(
            f"Image step is too small for encoding={encoding}: step={step}, "
            f"required_at_least={expected_visible_bytes}"
        )

    buffer = np.frombuffer(message.data, dtype=np.uint8)
    expected_total_bytes = height * step
    if buffer.size < expected_total_bytes:
        raise ValueError(
            f"Image buffer too small for declared dimensions: size={buffer.size}, "
            f"required={expected_total_bytes}"
        )

    rows = buffer[:expected_total_bytes].reshape(height, step)
    visible = rows[:, :expected_visible_bytes]

    if channels == 1:
        mono = visible.reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    image = visible.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def bgr_to_image_msg(image_bgr: np.ndarray, *, header: Any = None, image_cls: type[Any] | None = None) -> Any:
    """Convert OpenCV BGR/HWC/uint8 to a ROS Image message with bgr8 encoding."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got shape={image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got dtype={image_bgr.dtype}")

    if image_cls is None:
        from sensor_msgs.msg import Image  # type: ignore

        image_cls = Image

    contiguous = np.ascontiguousarray(image_bgr)
    message = image_cls()
    if header is not None:
        message.header = header
    message.height = int(contiguous.shape[0])
    message.width = int(contiguous.shape[1])
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = int(contiguous.shape[1] * 3)
    message.data = contiguous.tobytes()
    return message


def _channels_for_encoding(encoding: str) -> int:
    if encoding in {"bgr8", "rgb8"}:
        return 3
    if encoding in {"bgra8", "rgba8"}:
        return 4
    if encoding == "mono8":
        return 1
    raise ValueError(f"Unsupported image encoding: {encoding}")
