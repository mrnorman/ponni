from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_onnx_reference(
    model_path: str | Path,
    output_names: list[str],
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(output_names, inputs)
