from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from kokkos_nn.importer import import_onnx
from kokkos_nn.interpreter import run_graph


def _unary_model(path: Path, operation: str) -> Path:
    shape = [8, "batch"]
    graph = helper.make_graph(
        [helper.make_node(operation, ["input"], ["output"])],
        f"{operation.lower()}_semantics",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    for key, value in (("ponni.orientation", "features_batch"), ("ponni.batch_symbol", "batch")):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)
    return path


class MathOperatorTests(unittest.TestCase):
    def test_round_and_sign_match_onnx_edge_semantics(self) -> None:
        values = np.array([2.5, 1.5, -4.5, -3.5, 0.0, -0.0, np.nan, np.inf], dtype=np.float32).reshape(8, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for operation in ("Round", "Sign"):
                model_path = _unary_model(root / f"{operation.lower()}.onnx", operation)
                expected = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).run(
                    ["output"], {"input": values}
                )[0]
                actual = run_graph(import_onnx(model_path), values)
                np.testing.assert_equal(actual, expected)
                np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


if __name__ == "__main__":
    unittest.main()
