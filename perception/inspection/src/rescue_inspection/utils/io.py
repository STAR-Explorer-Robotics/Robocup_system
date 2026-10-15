"""Filesystem and artifact IO helpers."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from pydantic import BaseModel


def ensure_dir(path: Path) -> Path:
    """Create a directory when needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_payload(data: BaseModel | dict) -> dict:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    return data


def write_json(path: Path, data: BaseModel | dict) -> None:
    """Write JSON with indentation."""

    payload = _serialize_payload(data)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: Path, data: BaseModel | dict) -> None:
    """Append a JSON object to a jsonl file."""

    payload = _serialize_payload(data)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


class FFmpegVideoWriter:
    """Stream raw BGR frames into ffmpeg to produce an H.264 MP4."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.width, self.height = size
        self.fps = fps
        self._closed = False

        ensure_dir(path.parent)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to generate H.264 overlay videos.") from exc

        if self._process.stdin is None or self._process.stderr is None:
            raise RuntimeError(f"Failed to open ffmpeg video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        """Send a single frame to ffmpeg."""

        if self._closed:
            raise RuntimeError("Cannot write to a closed ffmpeg video writer.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 BGR images.")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"Video frame size mismatch: expected {(self.width, self.height)}, "
                f"got {(frame.shape[1], frame.shape[0])}."
            )
        if frame.dtype != np.uint8:
            raise ValueError("Video frames must use uint8 pixels.")

        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            message = self._read_stderr() or "ffmpeg closed the input pipe unexpectedly."
            raise RuntimeError(f"Failed while streaming overlay frames to ffmpeg: {message}") from exc

    def close(self, *, interrupted: bool = False, logger: logging.Logger | None = None) -> None:
        """Finish the ffmpeg process and flush the final MP4 file."""

        if self._closed:
            return
        self._closed = True

        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass

        return_code = self._process.wait()
        stderr = self._read_stderr()
        if return_code != 0:
            message = stderr or "ffmpeg exited with a non-zero status."
            if interrupted and _is_interrupted_ffmpeg_exit(return_code, stderr):
                if logger is not None:
                    logger.warning(
                        "Overlay video finalization was interrupted during user-requested shutdown; "
                        "partial video may still be usable: %s",
                        message,
                    )
                return
            raise RuntimeError(f"Failed to finalize H.264 overlay video: {message}")

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> FFmpegVideoWriter:
    """Create an ffmpeg-backed H.264 MP4 writer."""

    return FFmpegVideoWriter(path, fps, size)


def estimate_output_fps(
    previous_time_sec: float | None,
    current_time_sec: float | None,
    fallback_fps: float,
) -> float:
    """Estimate a stable output FPS from consecutive frame times."""

    if previous_time_sec is None or current_time_sec is None:
        return fallback_fps
    delta = current_time_sec - previous_time_sec
    if delta <= 0.0:
        return fallback_fps
    estimated_fps = 1.0 / delta
    if estimated_fps < 1.0 or estimated_fps > 240.0:
        return fallback_fps
    return estimated_fps


def _is_interrupted_ffmpeg_exit(return_code: int, stderr: str) -> bool:
    """Return whether ffmpeg appears to have exited due to an external interrupt."""

    normalized = stderr.lower()
    interrupt_markers = (
        "received signal",
        "signal 2",
        "signal 15",
        "immediate exit requested",
        "exiting normally",
    )
    if return_code < 0:
        return True
    if return_code in {130, 143, 255}:
        return any(marker in normalized for marker in interrupt_markers) or not normalized
    return any(marker in normalized for marker in interrupt_markers)
