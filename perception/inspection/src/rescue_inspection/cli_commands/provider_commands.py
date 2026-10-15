"""Provider inspection command handlers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rescue_inspection.logging_utils import configure_logging
from rescue_inspection.utils.onnx_runtime import create_onnx_session


def run_check_onnx_providers_command() -> dict[str, object]:
    """Print ONNX Runtime providers and run a tiny CUDA smoke test."""

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise RuntimeError("onnxruntime is unavailable in this container") from exc
    available_providers = list(ort.get_available_providers())
    payload: dict[str, object] = {
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "available_providers": available_providers,
    }
    payload["cuda_smoke"] = _run_onnx_cuda_smoke(ort, available_providers)
    return payload


def _run_onnx_cuda_smoke(ort: object, available_providers: list[str]) -> dict[str, object]:
    requested = [
        provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in available_providers
    ] or available_providers
    if not requested:
        return {"status": "skipped", "reason": "no_available_providers"}
    return {
        **_create_and_run_tiny_onnx_session(ort, requested),
        "requested_providers": requested,
    }


def _create_and_run_tiny_onnx_session(ort: object, requested_providers: list[str]) -> dict[str, object]:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        return {"status": "skipped", "reason": f"onnx_unavailable: {exc}"}

    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", inputs=["input"], outputs=["output"])],
        name="provider_smoke",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "provider_smoke.onnx"
        onnx.save(model, model_path)
        session, actual_providers = create_onnx_session(
            ort,
            str(model_path),
            requested_providers,
            logger=configure_logging("INFO"),
            context="onnx provider smoke",
        )
        outputs = session.run(None, {"input": np.asarray([[1.0]], dtype=np.float32)})
    return {
        "status": "ok",
        "actual_session_providers": actual_providers,
        "output": np.asarray(outputs[0]).tolist(),
    }

