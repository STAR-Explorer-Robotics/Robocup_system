"""Optional local OpenCV sink for live sensing overlays."""

from __future__ import annotations

import logging
import os
import sys

import cv2


class LocalOverlayWindow:
    """Display inspection-rendered overlay frames without going through ROS Image."""

    def __init__(self, name: str, wait_key_ms: int = 1, *, logger: logging.Logger | object | None = None) -> None:
        self.name = name
        self.wait_key_ms = max(int(wait_key_ms), 1)
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = True
        self._window_created = False
        self._warning_logged = False

    def show(self, image_bgr) -> None:
        """Show one BGR frame, disabling the sink if OpenCV GUI is unavailable."""

        if not self.enabled:
            return
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self._disable("DISPLAY/WAYLAND_DISPLAY is not set; local overlay window is disabled.")
            return
        try:
            if not self._window_created:
                cv2.namedWindow(self.name, opencv_window_flags())
                self._window_created = True
            cv2.imshow(self.name, image_bgr)
            cv2.waitKey(self.wait_key_ms)
        except cv2.error as exc:
            self._disable(f"OpenCV GUI failed for local overlay window; disabling it: {exc}")

    def close(self) -> None:
        """Close the OpenCV window if it was created."""

        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass
        finally:
            self._window_created = False

    def _disable(self, message: str) -> None:
        self.enabled = False
        if self._warning_logged:
            return
        warning = getattr(self.logger, "warning", None)
        if warning is not None:
            warning(message)
        self._warning_logged = True


def opencv_window_flags() -> int:
    """Use a resizable OpenCV window without the Qt toolbar when available."""

    return int(cv2.WINDOW_NORMAL) | int(getattr(cv2, "WINDOW_GUI_NORMAL", 0))
