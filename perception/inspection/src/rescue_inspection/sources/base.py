"""Base source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rescue_inspection.models import FramePacket


class FrameSource(ABC):
    """Abstract interface shared by all frame sources."""

    def __init__(self, source_type: str) -> None:
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        """Return the logical source type name."""

        return self._source_type

    @property
    def fps(self) -> float | None:
        """Return source FPS when available."""

        return None

    @abstractmethod
    def open(self) -> None:
        """Open the source and prepare it for reading."""

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Read the next frame packet or return None when exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the source."""

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

