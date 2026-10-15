"""Shared Hazmat classifier preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_hazmat_patch(
    patch_bgr: np.ndarray,
    *,
    input_size: int,
) -> dict[str, object]:
    """Prepare a rectified BGR patch for Hazmat CNN inference."""

    resized_bgr = cv2.resize(patch_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rgb_float = resized_rgb.astype(np.float32) / 255.0
    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)
    classifier_input_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

    return {
        "tensor": tensor,
        "resized_bgr": resized_bgr,
        "resized_rgb": resized_rgb,
        "classifier_input_bgr": classifier_input_bgr,
        "meta": {
            "source_patch_shape": list(patch_bgr.shape),
            "input_size": input_size,
            "resize_interpolation": "INTER_AREA",
            "runtime_patch_color_space": "BGR",
            "model_color_space": "RGB",
            "normalization": {
                "mean": IMAGENET_MEAN.tolist(),
                "std": IMAGENET_STD.tolist(),
            },
            "tensor_shape": list(tensor.shape),
            "classifier_input_note": "classifier_input.png is the resized RGB model input rendered for humans before normalization.",
        },
    }


def save_rgb_debug_image(path: Path, rgb_image: np.ndarray) -> None:
    """Save an RGB image via OpenCV without swapping colors in the output file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_image)


def save_bgr_debug_image(path: Path, bgr_image: np.ndarray) -> None:
    """Save a BGR image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr_image)


def polygon_bbox_crop(image_bgr: np.ndarray, polygon_points: list[tuple[float, float]]) -> np.ndarray:
    """Extract a bounding-box crop around a polygon from a BGR frame."""

    height, width = image_bgr.shape[:2]
    xs = [point[0] for point in polygon_points]
    ys = [point[1] for point in polygon_points]
    x1 = max(0, int(np.floor(min(xs))))
    y1 = max(0, int(np.floor(min(ys))))
    x2 = min(width, int(np.ceil(max(xs))))
    y2 = min(height, int(np.ceil(max(ys))))
    return image_bgr[y1:y2, x1:x2].copy()


def channel_statistics(image_bgr: np.ndarray) -> dict[str, object]:
    """Return lightweight color statistics for diagnostics."""

    if image_bgr.size == 0 or image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        return {
            "empty": True,
            "bgr_mean": [],
            "rgb_mean": [],
            "dominant_bgr_channel": "unknown",
            "weakest_bgr_channel": "unknown",
        }
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bgr_mean = image_bgr.reshape(-1, 3).mean(axis=0)
    rgb_mean = rgb.reshape(-1, 3).mean(axis=0)
    max_channel = ["blue", "green", "red"][int(np.argmax(bgr_mean))]
    min_channel = ["blue", "green", "red"][int(np.argmin(bgr_mean))]
    return {
        "bgr_mean": [float(value) for value in bgr_mean],
        "rgb_mean": [float(value) for value in rgb_mean],
        "dominant_bgr_channel": max_channel,
        "weakest_bgr_channel": min_channel,
    }
