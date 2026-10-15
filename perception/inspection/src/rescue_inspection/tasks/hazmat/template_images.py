"""Hazmat template image loading helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def hazmat_template_foreground_mask(template: np.ndarray) -> np.ndarray:
    """Estimate visible template foreground while ignoring transparent halos."""

    if template.ndim == 3 and template.shape[2] == 4:
        return template[..., 3] >= 240
    gray = cv2.cvtColor(template[..., :3], cv2.COLOR_BGR2GRAY)
    return gray < 248


def neutralize_template_alpha_halo(template: np.ndarray) -> np.ndarray:
    """Convert a Hazmat template to BGR with transparent edge RGB neutralized."""

    bgr = template[..., :3].copy()
    if template.ndim == 3 and template.shape[2] == 4:
        bgr[template[..., 3] < 240] = 255
    return bgr


def read_hazmat_template_bgr(path: Path | str) -> np.ndarray | None:
    """Read a Hazmat template as clean BGR, preserving alpha information during cleanup."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return neutralize_template_alpha_halo(image)
