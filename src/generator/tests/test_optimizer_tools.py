from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kokkos_nn.compiler import load_and_optimize, validate_model
from kokkos_nn.ir import DType, Graph, Node, Symbol, TensorValue
from kokkos_nn.planner import plan_storage
from kokkos_nn import planner


def _constant_expression_model(path: Path) -> Path:
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["one", "two"], ["three"]),
            helper.make_node("Add", ["input", "three"], ["output"]),
        ],
        "onnxscript_preprocess",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, "batch"])],
        [one, two],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    metadata = model.metadata_props.add()
    metadata.key = "ponni.orientation"
    metadata.value = "features_batch"
    onnx.save(model, path)
    return path


def _fragmented_arena_graph() -> Graph:
    """Four live intervals for which PONNI's original three greedy orders use 21 rather than 19 elements."""
    batch = Symbol("batch")
    tensors = {0: TensorValue(0, "input", (1, batch), DType.FLOAT32, is_input=True)}
    sizes = {1: 1, 2: 9, 3: 8, 4: 7, 5: 4, 6: 1, 7: 1, 8: 1}
    for tensor_id, size in sizes.items():
        tensors[tensor_id] = TensorValue(tensor_id, f"t{tensor_id}", (size, batch), DType.FLOAT32)
    for tensor_id in (1, 6, 7, 8):
        tensors[tensor_id].is_output = True
    nodes = [
        Node(0, "Dense", [0], [1]),
        Node(1, "Dense", [0], [2]),
        Node(2, "Dense", [2], [3]),
        Node(3, "Dense", [0], [4]),
        Node(4, "Dense", [0], [5]),
        Node(5, "Dense", [3, 5], [6]),
        Node(6, "Dense", [0], [7]),
        Node(7, "Dense", [4], [8]),
    ]
    graph = Graph([0], [1, 6, 7, 8], tensors, nodes, {})
    graph.rebuild_links()
    return graph


class OptimizerToolTests(unittest.TestCase):
    def test_onnxscript_preprocessing_is_optional_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _constant_expression_model(Path(directory) / "constant.onnx")
            _, without, without_report = load_and_optimize(model)
            original, with_preprocess, with_report = load_and_optimize(
                model, onnx_preprocess=True,
            )
            self.assertEqual([node.op for node in without.nodes], ["Add"])
            self.assertEqual([node.op for node in with_preprocess.nodes], ["Add"])
            preprocess = with_report[0]
            self.assertEqual(preprocess["name"], "onnxscript-preprocess")
            self.assertFalse(without_report[0]["changed"])
            self.assertTrue(preprocess["changed"])
            self.assertLess(preprocess["nodes_after"], preprocess["nodes_before"])
            self.assertEqual(len(original.nodes), 2)

    def test_native_exact_small_arena_closes_greedy_fragmentation_gap(self) -> None:
        graph = _fragmented_arena_graph()
        heuristic = plan_storage(graph, placement="heuristic")
        native = plan_storage(graph)
        exact = plan_storage(graph, placement="exact")
        self.assertEqual(heuristic.total_elements, 21)
        self.assertEqual(native.total_elements, 19)
        self.assertEqual(exact.total_elements, 19)
        self.assertEqual(native.placement_strategy, "exact-enumeration")
        self.assertTrue(native.optimality_proven)

    def test_workspace_oracle_compares_heuristic_native_and_exact_plans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _constant_expression_model(Path(directory) / "constant.onnx")
            report = validate_model(model, analyze_workspace=True)
            oracle = report["workspace_oracle"]
            self.assertIn("floating", oracle)
            self.assertIn("boolean", oracle)
            self.assertLessEqual(
                oracle["floating"]["native_elements"],
                oracle["floating"]["heuristic_elements"],
            )
            self.assertIn(
                oracle["floating"]["exact_backend"],
                {"exact-enumeration", "cp-sat", "cp-sat-feasible", "heuristic-exact-limit",
                 "heuristic-cp-sat-timeout"},
            )

    @unittest.skipUnless(importlib.util.find_spec("ortools"), "optional OR-Tools is not installed")
    def test_cp_sat_backend_proves_a_larger_layout(self) -> None:
        intervals = [(0, 9, index, index + 1, (index,)) for index in range(10)]
        result = planner._cp_sat_place(intervals)
        self.assertIsNotNone(result)
        elements, offsets, proven = result
        self.assertEqual(elements, 55)
        self.assertEqual(len(offsets), 10)
        self.assertTrue(proven)


if __name__ == "__main__":
    unittest.main()
