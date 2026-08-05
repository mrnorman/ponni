"""Unit tests for canonicalization, fusion, planning, and artifact emission.

Small synthetic ONNX graphs isolate individual compiler invariants; exported
operator-zoo coverage provides the broader end-to-end semantic check.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kokkos_nn.compiler import compile_model, load_and_optimize, validate_model
from kokkos_nn.errors import CompilerError
from kokkos_nn.export import export_operator_zoo
from kokkos_nn.interpreter import run_graph
from kokkos_nn.onnx_reference import run_onnx_reference
from kokkos_nn.planner import plan_storage
from kokkos_nn.scheduler import schedule_dense_chains
from kokkos_nn.weights import validate_weight_blob


def _metadata(model) -> None:
    for key, value in (("ponni.orientation", "features_batch"), ("ponni.batch_symbol", "batch")):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _save_model(path: Path, nodes, initializers, input_shape=(4, "batch"), output_shape=(3, "batch")) -> Path:
    graph = helper.make_graph(
        nodes,
        "test_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_shape))],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    _metadata(model)
    onnx.save(model, path)
    return path


def _gemm_model(path: Path, transposed_weight: bool = False) -> Path:
    weight_output_input = np.arange(12, dtype=np.float32).reshape(3, 4) / 17
    weight = weight_output_input if transposed_weight else weight_output_input.T
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    nodes = [
        helper.make_node("Transpose", ["input"], ["batch_input"], perm=[1, 0]),
        helper.make_node("Gemm", ["batch_input", "weight", "bias"], ["dense"],
                         transB=1 if transposed_weight else 0),
        helper.make_node("Tanh", ["dense"], ["activated"]),
        helper.make_node("Transpose", ["activated"], ["output"], perm=[1, 0]),
    ]
    initializers = [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")]
    return _save_model(path, nodes, initializers)


def _matmul_residual_model(path: Path, multiple_consumers: bool = False) -> Path:
    weight = (np.eye(4, dtype=np.float32) * 0.5).T
    bias = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    nodes = [
        helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
        helper.make_node("MatMul", ["x", "weight"], ["matmul"]),
        helper.make_node("Add", ["matmul", "bias"], ["dense"]),
        helper.make_node("Tanh", ["dense"], ["activated"]),
    ]
    residual_input = "dense" if multiple_consumers else "x"
    nodes.extend(
        [
            helper.make_node("Add", ["activated", residual_input], ["residual"]),
            helper.make_node("Sigmoid", ["residual"], ["result"]),
            helper.make_node("Transpose", ["result"], ["output"], perm=[1, 0]),
        ]
    )
    initializers = [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")]
    return _save_model(path, nodes, initializers, output_shape=(4, "batch"))


class CompilerTests(unittest.TestCase):
    """Exercise graph rewrites and the generated artifacts they influence."""
    def test_operator_zoo_matches_onnx_runtime_before_and_after_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = export_operator_zoo(root)
            original, optimized, _ = load_and_optimize(exported.model_path)
            operations = {node.op for node in optimized.nodes}

            def record_steps(steps) -> None:
                for step in steps:
                    operations.add(step["op"])
                    record_steps(step.get("attributes", {}).get("steps", []))

            for node in optimized.nodes:
                record_steps(node.attributes.get("steps", []))
                record_steps(node.attributes.get("map_steps", []))
            required = {
                "Abs", "Acos", "Acosh", "And", "Asin", "Asinh", "Atan", "Atanh", "BatchNormalization", "Cast",
                "Ceil", "Celu", "Clip", "CompareSelect", "Cos", "Cosh", "Elu", "Equal", "Erf", "Exp", "Floor",
                "Gather", "Gelu", "Greater", "GreaterOrEqual", "HardSigmoid", "HardSwish", "IsInf", "IsNaN",
                "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log", "LogSoftmax", "LpNormalization",
                "Mean", "Mish", "Neg", "Not", "Or", "PRelu", "ReduceL1", "ReduceL2", "ReduceLogSum",
                "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum",
                "ReduceSumSquare", "Round", "Selu", "Sign", "Silu", "Sin", "Sinh", "Softmax", "Softplus",
                "Softsign", "Sqrt", "Sum", "Tan", "ThresholdedRelu", "Where", "Xor",
            }
            self.assertTrue(required <= operations)
            original_only = {"CastLike", "Dropout", "Shape", "Size", "Squeeze", "Unsqueeze"}
            self.assertTrue(original_only <= set(original.metadata["operator_counts"]))
            self.assertNotIn("Sigmoid", operations)
            values = np.random.default_rng(44).standard_normal((8, 13)).astype(np.float32)
            reference = run_onnx_reference(exported.model_path, ["output"], {"input": values})[0]
            np.testing.assert_allclose(run_graph(original, values), reference, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(run_graph(optimized, values), reference, rtol=2e-6, atol=2e-6)

            report = compile_model(exported.model_path, root / "generated", model_name="OperatorZooModel")
            generated = (root / "generated" / "OperatorZooModel.hpp").read_text()
            manifest = json.loads((root / "generated" / "weights.json").read_text())
            self.assertIn("Scalar exponential_sum", generated)
            self.assertIn("Scalar second_moment", generated)
            self.assertIn("ponni::TwoHalf exponential_sum", generated)
            self.assertNotIn("ponni::TwoMask mask_workspace", generated)
            self.assertIn("ponni::TwoHalf::select", generated)
            self.assertNotIn("Kokkos::TeamPolicy", generated)
            self.assertNotIn("preactivation", generated)
            self.assertEqual(report["storage"]["external_workspace_bytes"], 0)
            self.assertEqual(report["sample_local_storage"]["mask_workspace_elements"], 0)
            self.assertEqual(report["onnx_opsets"]["ai.onnx"], 21)
            self.assertIn("Gelu:20", report["onnx_operator_schema_counts"])
            self.assertEqual(report["learned_parameter_count"], 33)
            self.assertEqual(manifest["learned_parameter_count"], 33)
            learned_names = {entry["name"] for entry in manifest["tensors"] if entry["learned"]}
            self.assertEqual(learned_names, {"bn_scale", "bn_bias", "ln_scale", "ln_bias", "prelu_slope"})

    def test_comparison_where_fusion_eliminates_mask_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Greater", ["x", "zero"], ["condition"]),
                helper.make_node("Neg", ["x"], ["negative"]),
                helper.make_node("Where", ["condition", "x", "negative"], ["selected"]),
                helper.make_node("Transpose", ["selected"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "compare_select.onnx",
                nodes,
                [numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")],
                output_shape=(4, "batch"),
            )
            fused = compile_model(model, root / "fused", model_name="CompareSelectModel")
            unfused = compile_model(
                model,
                root / "unfused",
                model_name="WhereModel",
                disabled_passes={"comparison-where-fusion", "pointwise-region-fusion"},
            )
            self.assertEqual(fused["optimized_operations"], ["PointwiseRegion"])
            self.assertIn("CompareSelect", fused["optimized_component_operations"])
            self.assertEqual(fused["sample_local_storage"]["mask_workspace_elements"], 0)
            self.assertEqual(unfused["sample_local_storage"]["mask_workspace_elements"], 4)
            self.assertIn("Where", unfused["optimized_operations"])
            generated = (root / "fused" / "CompareSelectModel.hpp").read_text()
            self.assertIn("ponni::TwoHalf::select(ponni::TwoHalf::greater", generated)

    def test_training_dropout_and_runtime_shape_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = numpy_helper.from_array(np.array(True, dtype=np.bool_), "training")
            dropout = _save_model(
                root / "training_dropout.onnx",
                [helper.make_node("Dropout", ["input", "", "training"], ["output"])],
                [training],
                input_shape=(4, "batch"),
                output_shape=(4, "batch"),
            )
            with self.assertRaisesRegex(CompilerError, "supports only inference mode"):
                compile_model(dropout, root / "dropout_out")

            runtime_shape = _save_model(
                root / "runtime_shape.onnx",
                [
                    helper.make_node("Shape", ["input"], ["runtime_shape"]),
                    helper.make_node("Identity", ["input"], ["output"]),
                ],
                [],
                input_shape=(4, "batch"),
                output_shape=(4, "batch"),
            )
            with self.assertRaisesRegex(CompilerError, "depends on the runtime batch size"):
                compile_model(runtime_shape, root / "shape_out")

    def test_activation_attributes_survive_dense_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight"], ["dense"]),
                helper.make_node("LeakyRelu", ["dense"], ["activated"], alpha=0.125),
                helper.make_node("Transpose", ["activated"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "leaky.onnx", nodes, [weight], output_shape=(4, "batch"))
            _, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["DenseBiasActivation"])
            self.assertEqual(optimized.nodes[0].attributes["activation_attributes"]["alpha"], 0.125)
            compile_model(model, root / "out", model_name="LeakyModel")
            generated = (root / "out" / "LeakyModel.hpp").read_text()
            self.assertIn("apply_leaky_relu(sum, static_cast<Scalar>(0.125))", generated)

    def test_rejects_reduction_or_softmax_over_batch_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            softmax = _save_model(
                root / "softmax_batch.onnx",
                [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
                [], output_shape=(4, "batch"),
            )
            with self.assertRaisesRegex(CompilerError, "static feature axis"):
                validate_model(softmax)

            axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
            reduction = _save_model(
                root / "reduce_batch.onnx",
                [helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=1)],
                [axes], output_shape=(4, 1),
            )
            with self.assertRaisesRegex(CompilerError, "feature-axis reduction"):
                validate_model(reduction)

    def test_gemm_transposed_weights_and_dense_activation_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for transposed in (False, True):
                model = _gemm_model(root / f"gemm_{transposed}.onnx", transposed)
                original, optimized, _ = load_and_optimize(model)
                self.assertEqual([node.op for node in optimized.nodes], ["DenseBiasActivation"])
                values = np.arange(28, dtype=np.float32).reshape(4, 7) / 13
                np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=1e-6, atol=1e-6)
                output = root / f"generated_{transposed}"
                report = compile_model(model, output, model_name="GemmModel")
                generated = (output / "GemmModel.hpp").read_text()
                self.assertIn("Scalar sum =", generated)
                self.assertNotIn("preactivation", generated)
                self.assertEqual(generated.count("void infer_one("), 1)
                self.assertEqual(generated.count("void infer_batch("), 1)
                self.assertEqual(generated.count("void infer_batch_half2("), 1)
                self.assertNotIn("infer_batch_hierarchical", generated)
                self.assertNotIn("infer_batch_team", generated)
                self.assertNotIn("Kokkos::TeamPolicy", generated)
                self.assertNotIn("Kokkos::LaunchBounds", generated)
                self.assertNotIn("team_shmem", generated)
                self.assertNotIn("infer_batch_half2_explicit", generated)
                self.assertIn("ponni::TwoHalf::fma", generated)
                self.assertEqual(
                    report["generated_targets"], ["infer_one", "infer_batch", "infer_batch_half2"]
                )
                self.assertEqual(report["optimized_operations"], ["DenseBiasActivation"])
                self.assertNotIn("autotuner_source", report)
                validate_weight_blob(output / "weights.ponni", json.loads((output / "weights.json").read_text()))

    def test_matmul_bias_residual_and_storage_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _matmul_residual_model(Path(directory) / "residual.onnx")
            original, optimized, _ = load_and_optimize(model)
            operations = [node.op for node in optimized.nodes]
            self.assertEqual(operations, ["DenseBiasActivation", "ResidualAddActivation"])
            plan = plan_storage(optimized)
            self.assertGreaterEqual(plan.total_elements, 4)
            values = np.random.default_rng(8).standard_normal((4, 32)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)
            report = compile_model(
                model, Path(directory) / "generated", model_name="ResidualModel",
            )
            self.assertEqual(report["generated_targets"], ["infer_one", "infer_batch", "infer_batch_half2"])
            generated = (Path(directory) / "generated" / "ResidualModel.hpp").read_text()
            self.assertNotIn("scratch_workspace", generated)
            self.assertNotIn("TeamThreadRange", generated)

    def test_two_dense_layers_stream_without_hidden_stack_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(19)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 64)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(64).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((64, 3)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(3).astype(np.float32), "bias1"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["hidden"]),
                helper.make_node("Gemm", ["hidden", "weight1", "bias1"], ["dense1"]),
                helper.make_node("Transpose", ["dense1"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "stream.onnx", nodes, initializers)
            report = compile_model(model, root / "out", model_name="StreamModel")
            generated = (root / "out" / "StreamModel.hpp").read_text()
            self.assertEqual(report["sample_local_storage"]["streamed_dense_pairs"], 1)
            self.assertEqual(report["workspace_reduction_aggressiveness"], 3)
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)
            self.assertIn("Scalar hidden =", generated)
            self.assertIn("Scalar output_accumulator_0", generated)
            self.assertNotIn("Scalar workspace[64]", generated)
            self.assertNotIn("sample_inputs", generated)
            self.assertNotIn("sample_outputs", generated)
            self.assertIn("ponni::SArray<Scalar,num_inputs> inputs;", generated)
            self.assertIn("inputs(i) = input_view(i,ibatch)", generated)
            direct_batch = generated.split("void infer_batch_half2(", 1)[0].rsplit("void infer_batch(", 1)[1]
            self.assertNotIn("inputs(j,ibatch)", direct_batch)
            self.assertEqual(report["half2"]["accumulator_type"], "one dependent FP16 chain")
            self.assertNotIn("infer_batch_half2_explicit", generated)


    def test_generalized_dense_chain_streaming_uses_weighted_nonoverlapping_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(37)
            widths = [4, 7, 5, 8, 6, 3]
            initializers = []
            nodes = [helper.make_node("Transpose", ["input"], ["x0"], perm=[1, 0])]
            current = "x0"
            for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
                weight_name = f"weight{index}"
                bias_name = f"bias{index}"
                dense_name = f"dense{index}" if index + 2 < len(widths) else "dense_output"
                output_name = f"hidden{index}" if index + 2 < len(widths) else dense_name
                initializers.extend([
                    numpy_helper.from_array(
                        rng.standard_normal((input_width, output_width)).astype(np.float32), weight_name
                    ),
                    numpy_helper.from_array(rng.standard_normal(output_width).astype(np.float32), bias_name),
                ])
                nodes.append(helper.make_node("Gemm", [current, weight_name, bias_name], [dense_name]))
                if index + 2 < len(widths):
                    nodes.append(helper.make_node("Tanh", [dense_name], [output_name]))
                current = output_name
            nodes.append(helper.make_node("Transpose", [current], ["output"], perm=[1, 0]))
            model = _save_model(
                root / "deep_stream.onnx",
                nodes,
                initializers,
                input_shape=(widths[0], "batch"),
                output_shape=(widths[-1], "batch"),
            )
            report = compile_model(model, root / "out", model_name="DeepStreamModel")
            generated = (root / "out" / "DeepStreamModel.hpp").read_text()
            _, optimized, _ = load_and_optimize(model)
            schedules = [schedule_dense_chains(optimized, level) for level in range(1, 4)]
            self.assertEqual([len(schedule.pair_by_consumer) for schedule in schedules], [0, 1, 2])
            self.assertEqual(report["sample_local_storage"]["streamed_dense_pairs"], 2)
            self.assertEqual(report["dense_chain_schedule"]["eliminated_elements"], 15)
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 11)
            self.assertEqual(
                report["dense_chain_schedule"]["decision_counts"],
                {"materialize": 2, "stream": 2, "retain": 0, "recompute": 0},
            )
            self.assertIn("Scalar workspace[11]", generated)

    def test_virtual_dense_input_is_not_misclassified_as_a_streaming_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(113)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 4)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((8, 2)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(2).astype(np.float32), "bias1"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["hidden0"]),
                helper.make_node("Tanh", ["hidden0"], ["hidden"]),
                helper.make_node("Concat", ["hidden", "x"], ["joined"], axis=1),
                helper.make_node("Gemm", ["joined", "weight1", "bias1"], ["result"]),
                helper.make_node("Transpose", ["result"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "virtual_stream.onnx", nodes, initializers, output_shape=(2, "batch")
            )
            original, optimized, _ = load_and_optimize(model)
            dense_consumer = optimized.nodes[1]
            self.assertIn("input_map", dense_consumer.attributes)
            schedule = schedule_dense_chains(optimized, 5)
            self.assertEqual(schedule.pair_by_consumer, {})
            values = rng.standard_normal((4, 7)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)

    def test_plain_linear_dense_pair_can_stream_without_an_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(163)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 7)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(7).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((7, 2)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(2).astype(np.float32), "bias1"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["hidden"]),
                helper.make_node("Gemm", ["hidden", "weight1", "bias1"], ["result"]),
                helper.make_node("Transpose", ["result"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "linear_stream.onnx", nodes, initializers, output_shape=(2, "batch"))
            _, optimized, _ = load_and_optimize(model)
            schedule = schedule_dense_chains(optimized, 3)
            self.assertEqual(len(schedule.pair_by_consumer), 1)
            report = compile_model(model, root / "out", model_name="LinearStreamModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)

    def test_workspace_reduction_aggressiveness_rejects_values_outside_one_through_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (0, 6):
                with self.assertRaisesRegex(
                    CompilerError, "--workspace-reduction-aggressiveness must be an integer from 1 through 5",
                ):
                    compile_model(
                        root / "unused.onnx", root / "out",
                        workspace_reduction_aggressiveness=value,
                    )

    def test_storage_reuse_after_last_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constants = [
                numpy_helper.from_array(np.array(2, dtype=np.float32), "two"),
                numpy_helper.from_array(np.array(1, dtype=np.float32), "one"),
                numpy_helper.from_array(np.array(3, dtype=np.float32), "three"),
            ]
            nodes = [
                helper.make_node("Mul", ["input", "two"], ["a"]),
                helper.make_node("Add", ["a", "one"], ["b"]),
                helper.make_node("Sub", ["input", "three"], ["c"]),
                helper.make_node("Add", ["b", "c"], ["output"]),
            ]
            model = _save_model(root / "reuse.onnx", nodes, constants, output_shape=(4, "batch"))
            _, optimized, _ = load_and_optimize(
                model, {"elementwise-chain-fusion", "pointwise-region-fusion"}
            )
            plan = plan_storage(optimized)
            self.assertGreaterEqual(plan.reused_tensors, 1)
            offsets = [slot.offset for slot in plan.slots.values()]
            self.assertLess(len(set(offsets)), len(offsets))

    def test_constant_folding_feeds_runtime_elementwise_operation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constants = [
                numpy_helper.from_array(np.array(2, dtype=np.float32), "two"),
                numpy_helper.from_array(np.array(3, dtype=np.float32), "three"),
            ]
            nodes = [
                helper.make_node("Add", ["two", "three"], ["five"]),
                helper.make_node("Div", ["input", "five"], ["output"]),
            ]
            model = _save_model(root / "fold.onnx", nodes, constants, output_shape=(4, "batch"))
            original, optimized, passes = load_and_optimize(model)
            constant_pass = next(item for item in passes if item["name"] == "constant-fold")
            self.assertTrue(constant_pass["changed"])
            values = np.arange(28, dtype=np.float32).reshape(4, 7)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=1e-6, atol=1e-6)

    def test_static_reshape_flatten_and_batch_transposes_fold_away(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = numpy_helper.from_array(np.array([-1, 2, 2], dtype=np.int64), "shape")
            nodes = [
                helper.make_node("Transpose", ["input"], ["batch_input"], perm=[1, 0]),
                helper.make_node("Reshape", ["batch_input", "shape"], ["reshaped"]),
                helper.make_node("Flatten", ["reshaped"], ["flat"], axis=1),
                helper.make_node("Transpose", ["flat"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "layout.onnx", nodes, [shape], output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual(optimized.nodes, [])
            values = np.arange(28, dtype=np.float32).reshape(4, 7)
            np.testing.assert_array_equal(run_graph(original, values), run_graph(optimized, values))

    def test_rejects_reshape_that_moves_batch_without_transposing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = numpy_helper.from_array(np.array([-1, 4], dtype=np.int64), "shape")
            nodes = [
                helper.make_node("Reshape", ["input", "shape"], ["batch_first"]),
                helper.make_node("Transpose", ["batch_first"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "unsafe_reshape.onnx", nodes, [shape], output_shape=(4, "batch")
            )
            with self.assertRaisesRegex(CompilerError, "batch-relative element order"):
                load_and_optimize(model)

    def test_identity_only_model_copies_input_to_every_generated_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _save_model(
                root / "identity.onnx",
                [helper.make_node("Identity", ["input"], ["output"])],
                [],
                output_shape=(4, "batch"),
            )
            original, optimized, _ = load_and_optimize(model)
            values = np.arange(28, dtype=np.float32).reshape(4, 7)
            np.testing.assert_array_equal(run_graph(original, values), run_graph(optimized, values))
            report = compile_model(model, root / "out", model_name="IdentityModel")
            generated = (root / "out" / "IdentityModel.hpp").read_text()
            self.assertEqual(report["optimized_operations"], [])
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)
            self.assertIn("outputs(i) = inputs(i);", generated)
            self.assertIn("output_view(i,ibatch) = outputs(i);", generated)
            self.assertIn("bool weights_loaded_ = storage_parameter_elements == 0;", generated)
            self.assertIn("GeneratedModel inference Views must use Kokkos::LayoutRight", generated)
            self.assertIn("infer_batch requires a nonzero batch size", generated)
            self.assertIn("infer_batch_half2 requires a nonzero batch size", generated)

    def test_static_feature_concat_imports_and_emits_for_all_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = numpy_helper.from_array(
                np.arange(24, dtype=np.float32).reshape(8, 3) / 29, "weight"
            )
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Concat", ["x", "x"], ["joined"], axis=1),
                helper.make_node("Gemm", ["joined", "weight"], ["dense"]),
                helper.make_node("Transpose", ["dense"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "concat.onnx", nodes, [weight])
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Dense"])
            self.assertEqual(len(optimized.nodes[0].attributes["input_map"]), 8)
            values = np.arange(28, dtype=np.float32).reshape(4, 7) / 11
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=1e-6, atol=1e-6)
            compile_model(model, root / "out", model_name="ConcatModel")
            generated = (root / "out" / "ConcatModel.hpp").read_text()
            self.assertNotIn("Scalar workspace[", generated)
            self.assertNotIn("ponni::TwoHalf workspace[", generated)
            self.assertIn("sum += parameters_", generated)
            self.assertIn("* inputs(", generated)

    def test_static_gather_feeds_dense_without_materializing_reordered_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indices = numpy_helper.from_array(np.array([3, 1, 1], dtype=np.int64), "indices")
            weight = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2) / 7, "weight")
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gather", ["x", "indices"], ["picked"], axis=1),
                helper.make_node("Gemm", ["picked", "weight"], ["dense"]),
                helper.make_node("Transpose", ["dense"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "gather_dense.onnx", nodes, [indices, weight], output_shape=(2, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Dense"])
            self.assertEqual(
                [entry["index"] for entry in optimized.nodes[0].attributes["input_map"]], [3, 1, 1]
            )
            values = np.random.default_rng(101).standard_normal((4, 5)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)
            report = compile_model(model, root / "out", model_name="GatherDenseModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)

    def test_nested_concat_and_gather_maps_compose_before_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_indices = numpy_helper.from_array(np.array([5, 1], dtype=np.int64), "left_indices")
            right_indices = numpy_helper.from_array(np.array([3, 0], dtype=np.int64), "right_indices")
            final_indices = numpy_helper.from_array(np.array([2, 1, 3], dtype=np.int64), "final_indices")
            weight = numpy_helper.from_array(np.arange(9, dtype=np.float32).reshape(3, 3) / 13, "weight")
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gather", ["x", "left_indices"], ["left"], axis=1),
                helper.make_node("Gather", ["x", "right_indices"], ["right"], axis=1),
                helper.make_node("Concat", ["left", "right"], ["joined"], axis=1),
                helper.make_node("Gather", ["joined", "final_indices"], ["picked"], axis=1),
                helper.make_node("Gemm", ["picked", "weight"], ["dense"]),
                helper.make_node("Transpose", ["dense"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "nested_virtual.onnx", nodes,
                [left_indices, right_indices, final_indices, weight],
                input_shape=(6, "batch"), output_shape=(3, "batch"),
            )
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Dense"])
            self.assertEqual(
                [entry["index"] for entry in optimized.nodes[0].attributes["input_map"]], [3, 1, 0]
            )
            values = np.random.default_rng(127).standard_normal((6, 5)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)

    def test_gather_of_dense_output_prunes_weight_and_bias_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indices = numpy_helper.from_array(np.array([3, 1, 1], dtype=np.int64), "indices")
            weight_values = np.arange(16, dtype=np.float32).reshape(4, 4) / 17
            bias_values = np.arange(4, dtype=np.float32) / 19
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight", "bias"], ["dense"], transB=1),
                helper.make_node("Gather", ["dense", "indices"], ["picked"], axis=1),
                helper.make_node("Transpose", ["picked"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "dense_gather.onnx", nodes,
                [numpy_helper.from_array(weight_values, "weight"),
                 numpy_helper.from_array(bias_values, "bias"), indices],
                output_shape=(3, "batch"),
            )
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Dense"])
            dense = optimized.nodes[0]
            np.testing.assert_array_equal(
                optimized.constants[optimized.tensors[dense.attributes["weight"]].constant_name].values,
                weight_values[[3, 1, 1], :],
            )
            values = np.random.default_rng(131).standard_normal((4, 6)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)

    def test_dense_residual_epilogue_eliminates_linear_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(103)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 6)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(6).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((6, 4)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "bias1"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["hidden"]),
                helper.make_node("Gemm", ["hidden", "weight1", "bias1"], ["linear"]),
                helper.make_node("Add", ["linear", "x"], ["residual"]),
                helper.make_node("Sigmoid", ["residual"], ["activated"]),
                helper.make_node("Transpose", ["activated"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "dense_residual.onnx", nodes, initializers, output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["DenseBiasActivation", "DenseResidualActivation"])
            values = rng.standard_normal((4, 7)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)
            report = compile_model(model, root / "out", model_name="DenseResidualModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)
            self.assertEqual(report["sample_local_storage"]["streamed_dense_pairs"], 1)

    def test_ordered_dense_batchnorm_activation_epilogue_fuses_without_reassociation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(133)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 4)).astype(np.float32), "weight"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "bias"),
                numpy_helper.from_array(rng.uniform(0.5, 1.5, 4).astype(np.float32), "scale"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "bn_bias"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "mean"),
                numpy_helper.from_array(rng.uniform(0.5, 1.5, 4).astype(np.float32), "variance"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight", "bias"], ["dense"]),
                helper.make_node(
                    "BatchNormalization", ["dense", "scale", "bn_bias", "mean", "variance"], ["normalized"]
                ),
                helper.make_node("Relu", ["normalized"], ["activated"]),
                helper.make_node("Transpose", ["activated"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(root / "dense_epilogue.onnx", nodes, initializers, output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["DenseEpilogue"])
            self.assertEqual(
                [step["op"] for step in optimized.nodes[0].attributes["epilogue_steps"]],
                ["BatchNormalization", "Relu"],
            )
            values = rng.standard_normal((4, 6)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)
            report = compile_model(model, root / "out", model_name="DenseEpilogueModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)

    def test_pointwise_map_streams_into_reduction_and_reuses_last_consumer_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offset = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "offset")
            axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
            nodes = [
                helper.make_node("Sub", ["input", "offset"], ["centered"]),
                helper.make_node("Mul", ["centered", "centered"], ["squared"]),
                helper.make_node("ReduceMean", ["squared", "axes"], ["output"], keepdims=1),
            ]
            model = _save_model(root / "mapped_reduce.onnx", nodes, [offset, axes], output_shape=(1, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["ReduceMean"])
            self.assertEqual([step["op"] for step in optimized.nodes[0].attributes["map_steps"]], ["Sub", "Mul"])
            values = np.random.default_rng(107).standard_normal((4, 9)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)
            report = compile_model(model, root / "out", model_name="MappedReductionModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)

    def test_reconverging_pointwise_region_streams_into_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offset = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "offset")
            axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
            nodes = [
                helper.make_node("Sub", ["input", "offset"], ["centered"]),
                helper.make_node("Mul", ["centered", "centered"], ["squared"]),
                helper.make_node("Abs", ["centered"], ["absolute"]),
                helper.make_node("Add", ["squared", "absolute"], ["mapped"]),
                helper.make_node("ReduceMean", ["mapped", "axes"], ["output"], keepdims=1),
            ]
            model = _save_model(root / "region_reduce.onnx", nodes, [offset, axes], output_shape=(1, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["ReduceMean"])
            self.assertIn("map_region_steps", optimized.nodes[0].attributes)
            values = np.random.default_rng(137).standard_normal((4, 9)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)
            report = compile_model(model, root / "out", model_name="RegionReductionModel")
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)

    def test_logical_predicate_where_and_activation_fuse_as_one_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = numpy_helper.from_array(np.array(-0.5, dtype=np.float32), "low")
            high = numpy_helper.from_array(np.array(0.75, dtype=np.float32), "high")
            zero = numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")
            nodes = [
                helper.make_node("Greater", ["input", "low"], ["above"]),
                helper.make_node("Less", ["input", "high"], ["below"]),
                helper.make_node("And", ["above", "below"], ["inside"]),
                helper.make_node("Where", ["inside", "input", "zero"], ["selected"]),
                helper.make_node("Relu", ["selected"], ["output"]),
            ]
            model = _save_model(root / "predicate_region.onnx", nodes, [low, high, zero], output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["PointwiseRegion"])
            values = np.random.default_rng(139).standard_normal((4, 8)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=0, atol=0)
            report = compile_model(model, root / "out", model_name="PredicateRegionModel")
            self.assertEqual(report["sample_local_storage"]["mask_workspace_elements"], 0)

    def test_stable_softmax_decomposition_canonicalizes_before_pointwise_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
            nodes = [
                helper.make_node("ReduceMax", ["input", "axes"], ["maximum"], keepdims=1),
                helper.make_node("Sub", ["input", "maximum"], ["shifted"]),
                helper.make_node("Exp", ["shifted"], ["exponential"]),
                helper.make_node("ReduceSum", ["exponential", "axes"], ["total"], keepdims=1),
                helper.make_node("Div", ["exponential", "total"], ["output"]),
            ]
            model = _save_model(root / "decomposed_softmax.onnx", nodes, [axes], output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Softmax"])
            values = np.random.default_rng(149).standard_normal((4, 7)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)

    def test_layernorm_decomposition_canonicalizes_before_region_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
            epsilon = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "epsilon")
            scale = numpy_helper.from_array(np.array([1.1, 0.9, 1.2, 0.8], dtype=np.float32), "scale")
            bias = numpy_helper.from_array(np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32), "bias")
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=1),
                helper.make_node("Sub", ["x", "mean"], ["centered"]),
                helper.make_node("Mul", ["centered", "centered"], ["square"]),
                helper.make_node("ReduceMean", ["square", "axes"], ["variance"], keepdims=1),
                helper.make_node("Add", ["variance", "epsilon"], ["stabilized"]),
                helper.make_node("Sqrt", ["stabilized"], ["deviation"]),
                helper.make_node("Div", ["centered", "deviation"], ["normalized"]),
                helper.make_node("Mul", ["normalized", "scale"], ["scaled"]),
                helper.make_node("Add", ["scaled", "bias"], ["result"]),
                helper.make_node("Transpose", ["result"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "decomposed_layernorm.onnx", nodes, [axes, epsilon, scale, bias], output_shape=(4, "batch")
            )
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["LayerNormalization"])
            values = np.random.default_rng(151).standard_normal((4, 8)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=3e-6, atol=3e-6)

    def test_mish_decomposition_canonicalizes_to_native_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = [
                helper.make_node("Softplus", ["input"], ["softplus"]),
                helper.make_node("Tanh", ["softplus"], ["tanh"]),
                helper.make_node("Mul", ["input", "tanh"], ["output"]),
            ]
            model = _save_model(root / "decomposed_mish.onnx", nodes, [], output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Mish"])
            values = np.random.default_rng(157).standard_normal((4, 5)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)

    def test_exact_gelu_decomposition_canonicalizes_to_native_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sqrt_two = numpy_helper.from_array(np.array(np.sqrt(2), dtype=np.float32), "sqrt_two")
            one = numpy_helper.from_array(np.array(1, dtype=np.float32), "one")
            half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half")
            nodes = [
                helper.make_node("Div", ["input", "sqrt_two"], ["scaled"]),
                helper.make_node("Erf", ["scaled"], ["erf"]),
                helper.make_node("Add", ["erf", "one"], ["offset"]),
                helper.make_node("Mul", ["input", "offset"], ["product"]),
                helper.make_node("Mul", ["product", "half"], ["output"]),
            ]
            model = _save_model(root / "decomposed_gelu.onnx", nodes, [sqrt_two, one, half], output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["Gelu"])
            values = np.random.default_rng(167).standard_normal((4, 5)).astype(np.float32)
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=2e-6, atol=2e-6)

    def test_multiple_consumers_prevent_illegal_dense_activation_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _matmul_residual_model(Path(directory) / "branched.onnx", multiple_consumers=True)
            _, optimized, _ = load_and_optimize(model)
            operations = [node.op for node in optimized.nodes]
            self.assertIn("Dense", operations)
            self.assertIn("Tanh", operations)

    def test_elementwise_chain_and_disableable_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scalar0 = numpy_helper.from_array(np.array(2, dtype=np.float32), "scale")
            scalar1 = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "offset")
            nodes = [
                helper.make_node("Mul", ["input", "scale"], ["scaled"]),
                helper.make_node("Add", ["scaled", "offset"], ["output"]),
            ]
            model = _save_model(root / "chain.onnx", nodes, [scalar0, scalar1], output_shape=(4, "batch"))
            _, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["ElementwiseChain"])
            _, unfused, _ = load_and_optimize(
                model, {"elementwise-chain-fusion", "pointwise-region-fusion"}
            )
            self.assertEqual([node.op for node in unfused.nodes], ["Mul", "Add"])

    def test_converging_elementwise_branches_are_owned_by_only_one_fused_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constants = [
                numpy_helper.from_array(np.array(value, dtype=np.float32), name)
                for name, value in (("shared_scale", 0.5), ("left_scale", 2.0), ("right_scale", -0.25))
            ]
            nodes = [
                helper.make_node("Mul", ["input", "shared_scale"], ["shared"]),
                helper.make_node("Mul", ["shared", "left_scale"], ["left"]),
                helper.make_node("Mul", ["shared", "right_scale"], ["right"]),
                helper.make_node("Add", ["left", "right"], ["output"]),
            ]
            model = _save_model(root / "converging.onnx", nodes, constants, output_shape=(4, "batch"))
            original, optimized, _ = load_and_optimize(model)
            self.assertEqual([node.op for node in optimized.nodes], ["PointwiseRegion"])
            for tensor in optimized.tensors.values():
                if tensor.is_constant or tensor.is_input:
                    continue
                if tensor.consumers or tensor.is_output:
                    self.assertIsNotNone(tensor.producer, tensor.name)
            values = np.random.default_rng(71).standard_normal((4, 7)).astype(np.float32)
            np.testing.assert_allclose(run_graph(optimized, values), run_graph(original, values), rtol=0, atol=0)

    def test_dead_node_elimination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = numpy_helper.from_array(np.array(1, dtype=np.float32), "one")
            nodes = [
                helper.make_node("Identity", ["input"], ["output"]),
                helper.make_node("Mul", ["input", "one"], ["dead"]),
            ]
            model = _save_model(root / "dead.onnx", nodes, [one], output_shape=(4, "batch"))
            report = validate_model(model)
            self.assertEqual(report["optimized_operations"], [])

    def test_rejects_dynamic_non_batch_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.onnx"
            node = helper.make_node("Identity", ["input"], ["output"])
            _save_model(path, [node], [], input_shape=("features", "batch"), output_shape=("features", "batch"))
            with self.assertRaisesRegex(CompilerError, "dynamic non-batch"):
                validate_model(path)

    def test_rejects_unsupported_operator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mod.onnx"
            divisor = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "divisor")
            _save_model(
                path, [helper.make_node("Mod", ["input", "divisor"], ["output"])], [divisor],
                output_shape=(4, "batch")
            )
            with self.assertRaisesRegex(CompilerError, "unsupported operator 'Mod'"):
                validate_model(path)

    def test_rejects_unsupported_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = numpy_helper.from_array(np.ones(2, dtype=np.float32), "bad")
            model = _save_model(
                root / "broadcast.onnx", [helper.make_node("Add", ["input", "bad"], ["output"])], [bad],
                output_shape=(4, "batch")
            )
            with self.assertRaisesRegex(CompilerError, "broadcast"):
                compile_model(model, root / "out")

    def test_rejects_variadic_min_max_until_codegen_supports_all_operands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower = numpy_helper.from_array(np.array(0, dtype=np.float32), "lower")
            upper = numpy_helper.from_array(np.array(1, dtype=np.float32), "upper")
            model = _save_model(
                root / "variadic_max.onnx",
                [helper.make_node("Max", ["input", "lower", "upper"], ["output"])],
                [lower, upper], output_shape=(4, "batch"),
            )
            with self.assertRaisesRegex(CompilerError, "exactly two inputs"):
                validate_model(model)

    def test_rejects_corrupt_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _gemm_model(root / "model.onnx")
            compile_model(model, root / "out")
            blob = bytearray((root / "out" / "weights.ponni").read_bytes())
            blob[-1] ^= 0x80
            corrupt = root / "corrupt.bin"
            corrupt.write_bytes(blob)
            with self.assertRaisesRegex(CompilerError, "checksum mismatch"):
                validate_weight_blob(corrupt)


if __name__ == "__main__":
    unittest.main()
