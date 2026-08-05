"""Small CPU-only ONNX Runtime adapter used by exporters and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_onnx_reference(
    model_path: str | Path,
    output_names: list[str],
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Evaluate selected outputs without accelerator-provider drift."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(output_names, inputs)
