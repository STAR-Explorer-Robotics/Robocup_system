"""Preview backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


PreviewAction = str | None


class PreviewBackend(ABC):
    """Interface for optional runtime preview implementations."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the preview backend is currently active."""

    @abstractmethod
    def show(self, frame: np.ndarray) -> None:
        """Display a frame."""

    @abstractmethod
    def poll_action(self) -> PreviewAction:
        """Return a normalized action requested by the user."""

    @abstractmethod
    def close(self) -> None:
        """Release preview resources."""


class NullPreviewBackend(PreviewBackend):
    """No-op preview backend for headless operation."""

    @property
    def enabled(self) -> bool:
        return False

    def show(self, frame: np.ndarray) -> None:  # pragma: no cover - trivial
        return None

    def poll_action(self) -> PreviewAction:  # pragma: no cover - trivial
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None
