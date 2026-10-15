"""Geometry helpers for ROI detector preprocessing and postprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from rescue_inspection.models import Point2D
from rescue_inspection.tasks.vision_target.types import LetterboxInfo


def letterbox_image(image: np.ndarray, input_size: int, *, fill_value: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize an image into a square model canvas while preserving aspect ratio."""

    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), fill_value, dtype=image.dtype)
    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    x1 = int(round(pad_x))
    y1 = int(round(pad_y))
    canvas[y1:y1 + resized_h, x1:x1 + resized_w] = resized
    return canvas, LetterboxInfo(
        original_shape=(height, width),
        input_size=input_size,
        scale=scale,
        pad_x=float(x1),
        pad_y=float(y1),
        resized_shape=(resized_h, resized_w),
    )


def map_box_from_letterbox(
    bbox_xyxy: tuple[float, float, float, float],
    info: LetterboxInfo,
) -> tuple[float, float, float, float]:
    """Map an xyxy box from model input coordinates back to the original image."""

    x1, y1, x2, y2 = bbox_xyxy
    height, width = info.original_shape
    mapped = (
        (x1 - info.pad_x) / info.scale,
        (y1 - info.pad_y) / info.scale,
        (x2 - info.pad_x) / info.scale,
        (y2 - info.pad_y) / info.scale,
    )
    return clip_box(mapped, width=width, height=height)


def clip_box(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clip an xyxy box to image bounds."""

    x1, y1, x2, y2 = bbox_xyxy
    return (
        float(np.clip(x1, 0, max(width - 1, 0))),
        float(np.clip(y1, 0, max(height - 1, 0))),
        float(np.clip(x2, 0, max(width - 1, 0))),
        float(np.clip(y2, 0, max(height - 1, 0))),
    )


def xywh_to_xyxy(row: np.ndarray) -> tuple[float, float, float, float]:
    """Convert center xywh to xyxy."""

    cx, cy, width, height = [float(v) for v in row[:4]]
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Return IoU for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def nms(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float,
) -> list[int]:
    """Class-agnostic non-maximum suppression."""

    order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [index for index in order if box_iou(boxes[current], boxes[index]) <= iou_threshold]
    return kept


def crop_box(image: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Crop an xyxy box from an image."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(bbox_xyxy, width=width, height=height)
    ix1, iy1, ix2, iy2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    ix2 = max(ix1 + 1, min(ix2, width))
    iy2 = max(iy1 + 1, min(iy2, height))
    return image[iy1:iy2, ix1:ix2].copy()


def square_padded_crop(
    image: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    pad_ratio: float = 0.12,
) -> np.ndarray:
    """Crop a square region around a bbox with extra padding, clipped to the image."""

    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + pad_ratio * 2.0)
    square = (cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)
    return crop_box(image, square)


def polygon_from_box(bbox_xyxy: tuple[float, float, float, float]) -> list[Point2D]:
    """Create a clockwise polygon from an xyxy box."""

    x1, y1, x2, y2 = bbox_xyxy
    return [
        Point2D(x=x1, y=y1),
        Point2D(x=x2, y=y1),
        Point2D(x=x2, y=y2),
        Point2D(x=x1, y=y2),
    ]
