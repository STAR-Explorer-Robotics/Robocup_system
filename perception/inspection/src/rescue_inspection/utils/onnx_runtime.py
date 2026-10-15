"""ONNX Runtime provider selection helpers."""

from __future__ import annotations

import logging
from typing import Any


GPU_EXECUTION_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}


def preload_onnxruntime_gpu_dependencies(
    ort_module: Any,
    requested_providers: list[str],
    *,
    logger: logging.Logger,
    context: str,
) -> None:
    """Ask ONNX Runtime to preload CUDA/cuDNN libraries when GPU providers are requested."""

    if not GPU_EXECUTION_PROVIDERS.intersection(requested_providers):
        return
    preload = getattr(ort_module, "preload_dlls", None)
    if not callable(preload):
        return
    try:
        preload(directory="")
    except Exception:
        logger.exception("%s failed to preload ONNX Runtime GPU dependencies.", context)


def select_onnx_providers(
    ort_module: Any,
    requested_providers: list[str],
    *,
    logger: logging.Logger,
    context: str,
) -> list[str]:
    """Return available ONNX providers in requested priority order."""

    available = set(ort_module.get_available_providers())
    selected = [provider for provider in requested_providers if provider in available]
    missing = [provider for provider in requested_providers if provider not in available]
    if missing:
        logger.warning(
            "%s requested unavailable ONNX Runtime providers %s; available providers are %s.",
            context,
            missing,
            sorted(available),
        )
    if selected:
        return selected
    if "CPUExecutionProvider" in available:
        logger.warning(
            "%s falling back to CPUExecutionProvider because none of the requested providers are available.",
            context,
        )
        return ["CPUExecutionProvider"]
    raise RuntimeError(
        f"{context} has no usable ONNX Runtime provider. "
        f"requested={requested_providers}, available={sorted(available)}"
    )


def get_session_providers(session: Any, selected_providers: list[str]) -> list[str]:
    """Return actual providers reported by a session, or the selected request list."""

    get_providers = getattr(session, "get_providers", None)
    if callable(get_providers):
        return list(get_providers())
    return list(selected_providers)


def create_onnx_session(
    ort_module: Any,
    model_path: str,
    requested_providers: list[str],
    *,
    logger: logging.Logger,
    context: str,
) -> tuple[Any, list[str]]:
    """Create an ONNX Runtime session with provider fallback diagnostics."""

    preload_onnxruntime_gpu_dependencies(
        ort_module,
        requested_providers,
        logger=logger,
        context=context,
    )
    providers = select_onnx_providers(
        ort_module,
        requested_providers,
        logger=logger,
        context=context,
    )
    try:
        session = ort_module.InferenceSession(model_path, providers=providers)
    except Exception:
        if providers != ["CPUExecutionProvider"] and "CPUExecutionProvider" in ort_module.get_available_providers():
            logger.exception(
                "%s failed to initialize requested ONNX Runtime providers %s; retrying with CPUExecutionProvider.",
                context,
                providers,
            )
            providers = ["CPUExecutionProvider"]
            session = ort_module.InferenceSession(model_path, providers=providers)
        else:
            raise
    return session, get_session_providers(session, providers)
