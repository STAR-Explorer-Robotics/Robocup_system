"""OpenCV window preview backend."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from rescue_inspection.preview.base import PreviewAction, PreviewBackend


def _decode_key(key_code: int) -> PreviewAction:
    key_code &= 0xFF
    if key_code in {ord("q"), 27}:
        return "quit"
    return None


def _window_flags() -> int:
    """Return OpenCV window flags for a resizable preview without the toolbar."""

    return cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)


class OpenCVPreviewBackend(PreviewBackend):
    """Display frames in an OpenCV window and capture quit commands."""

    def __init__(self, window_name: str, logger: logging.Logger, wait_key_delay_ms: int = 1) -> None:
        self.window_name = window_name
        self.logger = logger
        self.wait_key_delay_ms = wait_key_delay_ms
        self._window_created = False
        self._enabled = True
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def show(self, frame: np.ndarray) -> None:
        if not self._enabled:
            return
        try:
            if not self._window_created:
                cv2.namedWindow(self.window_name, _window_flags())
                self._window_created = True
            cv2.imshow(self.window_name, frame)
        except Exception as exc:  # pragma: no cover - depends on local GUI backend
            self._disable_with_warning(exc)

    def poll_action(self) -> PreviewAction:
        if not self._enabled:
            return None
        try:
            return _decode_key(cv2.waitKey(self.wait_key_delay_ms))
        except Exception as exc:  # pragma: no cover - depends on local GUI backend
            self._disable_with_warning(exc)
            return None

    def close(self) -> None:
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:  # pragma: no cover - defensive cleanup
            pass
        self._window_created = False

    def _disable_with_warning(self, exc: Exception) -> None:
        self._enabled = False
        if self._warned:
            return
        self._warned = True
        self.logger.warning(
            "Preview backend is unavailable; continuing without live preview: %s",
            exc,
        )
