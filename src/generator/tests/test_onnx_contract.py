from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kokkos_nn.compiler import validate_model
from kokkos_nn.errors import CompilerError
from kokkos_nn.export import _normalize_feature_batch_boundaries
from kokkos_nn.importer import import_onnx
from kokkos_nn.interpreter import run_graph


def _metadata(model) -> None:
    for key, value in (("ponni.orientation", "features_batch"), ("ponni.batch_symbol", "batch")):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _save(path: Path, nodes, initializers=(), *, opset=18, input_shape=(4, "batch"),
          output_shape=(4, "batch"), dtype=TensorProto.FLOAT, extra_opsets=()) -> Path:
    graph = helper.make_graph(
        nodes,
        "onnx_contract",
        [helper.make_tensor_value_info("input", dtype, list(input_shape))],
        [helper.make_tensor_value_info("output", dtype, list(output_shape))],
        list(initializers),
    )
    imports = [helper.make_opsetid("", opset), *extra_opsets]
    model = helper.make_model(graph, opset_imports=imports)
    _metadata(model)
    onnx.save(model, path)
    return path


class OnnxContractTests(unittest.TestCase):
    def test_export_normalizes_emitter_chosen_batch_symbols(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "emitter_symbols",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, "s0"])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, "s0"])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        _normalize_feature_batch_boundaries(model, 4, "test exporter")
        self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[1].dim_param, "batch")
        self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[1].dim_param, "batch")

    def test_records_domain_opsets_schema_versions_and_materialized_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _save(
                Path(directory) / "defaults.onnx",
                [
                    helper.make_node("Transpose", ["input"], ["transposed"]),
                    helper.make_node("Transpose", ["transposed"], ["output"]),
                ],
                extra_opsets=(helper.make_opsetid("com.example.unused", 7),),
            )
            graph = import_onnx(model)
            self.assertEqual(graph.metadata["ir_version"], onnx.IR_VERSION)
            self.assertEqual(graph.metadata["opsets"], {"ai.onnx": 18, "com.example.unused": 7})
            self.assertEqual(graph.metadata["operator_schema_counts"], {"Transpose:13": 2})
            self.assertEqual([node.attributes["perm"] for node in graph.nodes], [[1, 0], [1, 0]])

    def test_rejects_onnx_ir_and_opsets_outside_reviewed_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for opset in (12, 23):
                path = _save(root / f"opset_{opset}.onnx", [helper.make_node("Identity", ["input"], ["output"])],
                             opset=opset)
                with self.assertRaisesRegex(CompilerError, f"unsupported ai.onnx opset {opset}"):
                    import_onnx(path)

            for ir_version in (7, 14):
                path = _save(root / f"ir_{ir_version}.onnx", [helper.make_node("Identity", ["input"], ["output"])])
                model = onnx.load(path)
                model.ir_version = ir_version
                onnx.save(model, path)
                with self.assertRaisesRegex(CompilerError, f"unsupported ONNX IR version {ir_version}"):
                    import_onnx(path)

    def test_clip_optional_input_hole_preserves_maximum_semantics(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            maximum = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "maximum")
            path = _save(
                Path(directory) / "clip_max.onnx",
                [helper.make_node("Clip", ["input", "", "maximum"], ["output"])],
                [maximum],
            )
            graph = import_onnx(path)
            self.assertEqual(len(graph.nodes[0].inputs), 1)
            self.assertNotIn("min", graph.nodes[0].attributes)
            self.assertEqual(graph.nodes[0].attributes["max"], 0.25)
            values = np.array([[-1.0, 1.0], [0.5, -0.5], [2.0, 0.1], [-3.0, 3.0]], dtype=np.float32)
            expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
                ["output"], {"input": values}
            )[0]
            np.testing.assert_array_equal(run_graph(graph, values), expected)

    def test_reduction_axes_normalize_across_schema_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = _save(
                root / "reduce_13.onnx",
                [helper.make_node("ReduceMean", ["input"], ["output"], axes=[0], keepdims=1)],
                opset=13,
                output_shape=(1, "batch"),
            )
            axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
            new_path = _save(
                root / "reduce_18.onnx",
                [helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=1)],
                [axes],
                opset=18,
                output_shape=(1, "batch"),
            )
            old_graph = import_onnx(old_path)
            new_graph = import_onnx(new_path)
            self.assertEqual(len(old_graph.nodes[0].inputs), 1)
            self.assertEqual(len(new_graph.nodes[0].inputs), 1)
            self.assertEqual(old_graph.nodes[0].attributes, new_graph.nodes[0].attributes)
            values = np.arange(20, dtype=np.float32).reshape(4, 5)
            np.testing.assert_array_equal(run_graph(old_graph, values), run_graph(new_graph, values))

    def test_semantic_restrictions_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = numpy_helper.from_array(np.array([4, -1], dtype=np.int64), "shape")
            reshape = _save(
                root / "allowzero.onnx",
                [helper.make_node("Reshape", ["input", "shape"], ["output"], allowzero=1)],
                [shape],
            )
            with self.assertRaisesRegex(CompilerError, "supports only allowzero=0"):
                validate_model(reshape)

            integer = _save(
                root / "integer.onnx",
                [helper.make_node("Identity", ["input"], ["output"])],
                dtype=TensorProto.INT32,
            )
            with self.assertRaisesRegex(CompilerError, "unsupported dtype int32"):
                validate_model(integer)


if __name__ == "__main__":
    unittest.main()
