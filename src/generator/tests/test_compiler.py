from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kokkos_nn.compiler import compile_model, load_and_optimize, validate_model
from kokkos_nn.emitter import half2_accumulator_heuristic
from kokkos_nn.errors import CompilerError
from kokkos_nn.export import export_operator_zoo
from kokkos_nn.interpreter import run_graph
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
    def test_half2_accumulator_heuristic_uses_conservative_cross_vendor_policy(self) -> None:
        self.assertEqual(half2_accumulator_heuristic(1), 0)
        self.assertEqual(half2_accumulator_heuristic(8), 0)
        self.assertEqual(half2_accumulator_heuristic(32), 0)
        self.assertEqual(half2_accumulator_heuristic(128, 3), 0)
        self.assertEqual(half2_accumulator_heuristic(128, 4), 0)
        self.assertEqual(half2_accumulator_heuristic(128, 8), 0)

    def test_operator_zoo_matches_onnx_runtime_before_and_after_optimization(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = export_operator_zoo(root)
            original, optimized, _ = load_and_optimize(exported.model_path)
            operations = {node.op for node in optimized.nodes}
            required = {
                "Abs", "BatchNormalization", "Clip", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish",
                "LayerNormalization", "LeakyRelu", "Log", "LogSoftmax", "Mish", "Neg", "ReduceMean",
                "ReduceSum", "Silu", "Softmax", "Softplus", "Sqrt",
            }
            self.assertTrue(required <= operations)
            self.assertNotIn("Sigmoid", operations)
            values = np.random.default_rng(44).standard_normal((8, 13)).astype(np.float32)
            reference = ort.InferenceSession(
                str(exported.model_path), providers=["CPUExecutionProvider"]
            ).run(["output"], {"input": values})[0]
            np.testing.assert_allclose(run_graph(original, values), reference, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(run_graph(optimized, values), reference, rtol=2e-6, atol=2e-6)

            report = compile_model(exported.model_path, root / "generated", model_name="OperatorZooModel")
            generated = (root / "generated" / "OperatorZooModel.hpp").read_text()
            manifest = json.loads((root / "generated" / "weights.json").read_text())
            self.assertIn("Scalar exponential_sum", generated)
            self.assertIn("Scalar second_moment", generated)
            self.assertIn("ponni::TwoHalf exponential_sum", generated)
            self.assertIn("TeamThreadRange(team, active_batch)", generated)
            self.assertNotIn("preactivation", generated)
            self.assertEqual(report["storage"]["external_workspace_bytes"], 0)
            self.assertEqual(report["learned_parameter_count"], 32)
            self.assertEqual(manifest["learned_parameter_count"], 32)
            learned_names = {entry["name"] for entry in manifest["tensors"] if entry["learned"]}
            self.assertEqual(learned_names, {"bn_scale", "bn_bias", "ln_scale", "ln_bias"})

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
                report = compile_model(model, output, strategy="sample-local", model_name="GemmModel")
                generated = (output / "GemmModel.hpp").read_text()
                self.assertIn("Scalar sum =", generated)
                self.assertNotIn("preactivation", generated)
                self.assertEqual(generated.count("void infer_one("), 1)
                self.assertEqual(generated.count("void infer_batch("), 1)
                self.assertEqual(generated.count("void infer_batch_hierarchical("), 1)
                self.assertEqual(generated.count("void infer_batch_tensorcore("), 1)
                self.assertEqual(generated.count("void infer_batch_half2("), 1)
                self.assertEqual(generated.count("void infer_batch_half2_heuristic("), 1)
                self.assertNotIn("void infer_batch_half2_explicit(", generated)
                self.assertIn("ponni::TwoHalf::fma", generated)
                self.assertIn("HalfParameterView", generated)
                self.assertIn("int const ibatch = 2 * ipair", generated)
                self.assertIn("Kokkos::TeamThreadRange(team", generated)
                self.assertIn("team.team_shmem().get_shmem", generated)
                self.assertIn("int const local_batch = linear % active_batch", generated)
                self.assertIn("int const i = linear / active_batch", generated)
                self.assertIn("int const batch_begin = team.league_rank() * batch_tile", generated)
                self.assertIn("default_hierarchical_batch_tile", generated)
                self.assertEqual(report["hierarchical_batch_tiling"]["index_order"],
                                 "linear = neuron * active_batch + local_batch")
                self.assertEqual(report["optimized_operations"], ["DenseBiasActivation"])
                validate_weight_blob(output / "weights.bin", json.loads((output / "weights.json").read_text()))

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
            self.assertTrue(report["sample_local_storage"]["streaming_dense_pair"])
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 0)
            self.assertIn("Scalar hidden =", generated)
            self.assertIn("Scalar output_accumulator_0", generated)
            self.assertNotIn("Scalar workspace[64]", generated)
            self.assertNotIn("sample_inputs", generated)
            self.assertNotIn("sample_outputs", generated)
            self.assertIn("ponni::SArray<Scalar,num_inputs> inputs;", generated)
            self.assertIn("inputs(i) = input_view(i,ibatch)", generated)
            direct_batch = generated.split("void infer_batch_hierarchical(", 1)[0].rsplit("void infer_batch(", 1)[1]
            self.assertNotIn("inputs(j,ibatch)", direct_batch)
            self.assertTrue(report["tensorcore"]["eligible"])
            self.assertEqual(report["tensorcore"]["launch"], "raw CUDA kernel (no Kokkos execution policy)")
            self.assertEqual(report["tensorcore"]["shared_memory_bytes_per_warp"], 2048)
            self.assertIn("bool static constexpr tensorcore_eligible = true", generated)
            self.assertIn("nvcuda::wmma::mma_sync", generated)
            self.assertIn("hidden_begin += 16", generated)
            self.assertIn("static __global__ void StreamModel_tensorcore_kernel", generated)
            self.assertIn("<<<block_count,thread_count,scratch_bytes,execution.cuda_stream()>>>", generated)
            self.assertNotIn("TensorCoreFunctor", generated)
            self.assertIn("void infer_batch_tensorcore(", generated)
            tensorcore_report = compile_model(
                model, root / "tensorcore", strategy="tensorcore", model_name="TensorCoreModel"
            )
            self.assertEqual(tensorcore_report["recommended_batched_target"], "infer_batch_tensorcore")
            half2_report = compile_model(model, root / "half2", strategy="half2", model_name="Half2Model")
            self.assertEqual(half2_report["recommended_batched_target"], "infer_batch_half2_heuristic")
            self.assertEqual(
                [entry["accumulators"] for entry in half2_report["half2"]["heuristic"]],
                [0, 0],
            )
            explicit_report = compile_model(
                model,
                root / "half2_explicit",
                strategy="half2",
                model_name="ExplicitHalf2Model",
                half2_accumulators="2,16",
            )
            explicit_generated = (root / "half2_explicit" / "ExplicitHalf2Model.hpp").read_text()
            self.assertIn("void infer_batch_half2_explicit(", explicit_generated)
            self.assertIn("output_accumulator_0_15", explicit_generated)
            self.assertEqual(
                [entry["accumulators"] for entry in explicit_report["half2"]["explicit"]],
                [2, 16],
            )
            for accumulator_count in (0, 2, 4, 8, 16, 32):
                policy_report = compile_model(
                    model,
                    root / f"half2_policy_{accumulator_count}",
                    half2_accumulators=accumulator_count,
                )
                self.assertEqual(
                    [entry["accumulators"] for entry in policy_report["half2"]["explicit"]],
                    [accumulator_count, accumulator_count],
                )
            with self.assertRaisesRegex(CompilerError, "provided 3 counts for 2 canonical dense nodes"):
                compile_model(model, root / "bad_half2_length", half2_accumulators="2,4,8")
            with self.assertRaisesRegex(CompilerError, "unsupported half2 accumulator count 3"):
                compile_model(model, root / "bad_half2_value", half2_accumulators=3)

    def test_three_dense_tensorcore_chain_uses_width_dependent_shared_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(31)
            width = 16
            initializers = [
                numpy_helper.from_array(rng.standard_normal((width, width)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(width).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((width, width)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(width).astype(np.float32), "bias1"),
                numpy_helper.from_array(rng.standard_normal((width, 3)).astype(np.float32), "weight2"),
                numpy_helper.from_array(rng.standard_normal(3).astype(np.float32), "bias2"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["hidden0"]),
                helper.make_node("Gemm", ["hidden0", "weight1", "bias1"], ["dense1"]),
                helper.make_node("Tanh", ["dense1"], ["hidden1"]),
                helper.make_node("Gemm", ["hidden1", "weight2", "bias2"], ["dense2"]),
                helper.make_node("Transpose", ["dense2"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "triple.onnx", nodes, initializers,
                input_shape=(width, "batch"), output_shape=(3, "batch")
            )
            report = compile_model(model, root / "out", strategy="tensorcore", model_name="TripleModel")
            generated = (root / "out" / "TripleModel.hpp").read_text()
            self.assertTrue(report["tensorcore"]["eligible"])
            self.assertEqual(report["tensorcore"]["dense_layers"], 3)
            self.assertEqual(report["tensorcore"]["shared_memory_bytes_per_warp"], 3584)
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], width)
            self.assertTrue(report["sample_local_storage"]["streaming_dense_tail"])
            self.assertIn("float * first_tile", generated)
            self.assertIn("for (int input_begin = 0; input_begin < padded_inputs; input_begin += 8)", generated)
            self.assertIn("for (int second_begin = 0; second_begin < 16; second_begin += 16)", generated)

    def test_tensorcore_rejects_three_dense_chain_above_shared_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            width = 384
            rng = np.random.default_rng(31)
            initializers = [
                numpy_helper.from_array(
                    rng.standard_normal((width, width)).astype(np.float32), "weight0"
                ),
                numpy_helper.from_array(np.zeros(width, dtype=np.float32), "bias0"),
                numpy_helper.from_array(
                    rng.standard_normal((width, 8)).astype(np.float32), "weight1"
                ),
                numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bias1"),
                numpy_helper.from_array(
                    rng.standard_normal((8, 3)).astype(np.float32), "weight2"
                ),
                numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias2"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["hidden0"]),
                helper.make_node("Gemm", ["hidden0", "weight1", "bias1"], ["dense1"]),
                helper.make_node("Tanh", ["dense1"], ["hidden1"]),
                helper.make_node("Gemm", ["hidden1", "weight2", "bias2"], ["dense2"]),
                helper.make_node("Transpose", ["dense2"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "large_tensorcore.onnx", nodes, initializers,
                input_shape=(width, "batch"), output_shape=(3, "batch"),
            )
            with self.assertRaisesRegex(CompilerError, "shared memory per warp"):
                compile_model(model, root / "out", strategy="tensorcore")

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
            self.assertEqual(report["sample_local_storage"]["streamed_dense_pairs"], 2)
            self.assertEqual(report["dense_chain_schedule"]["eliminated_elements"], 15)
            self.assertEqual(report["sample_local_storage"]["workspace_elements"], 6)
            self.assertEqual(
                report["dense_chain_schedule"]["decision_counts"],
                {"materialize": 2, "stream": 2, "retain": 0, "recompute": 0},
            )
            self.assertNotIn("workspace[13]", generated)

    def test_small_terminal_dense_branch_recomputes_only_under_explicit_cost_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(41)
            initializers = [
                numpy_helper.from_array(rng.standard_normal((4, 4)).astype(np.float32), "weight0"),
                numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "bias0"),
                numpy_helper.from_array(rng.standard_normal((4, 2)).astype(np.float32), "weight1"),
                numpy_helper.from_array(rng.standard_normal(2).astype(np.float32), "bias1"),
                numpy_helper.from_array(rng.standard_normal((4, 2)).astype(np.float32), "weight2"),
                numpy_helper.from_array(rng.standard_normal(2).astype(np.float32), "bias2"),
            ]
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["shared"]),
                helper.make_node("Gemm", ["shared", "weight1", "bias1"], ["left"]),
                helper.make_node("Gemm", ["shared", "weight2", "bias2"], ["right"]),
                helper.make_node("Add", ["left", "right"], ["joined"]),
                helper.make_node("Transpose", ["joined"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "recompute.onnx", nodes, initializers,
                input_shape=(4, "batch"), output_shape=(2, "batch")
            )
            recomputed = compile_model(
                model, root / "recomputed", model_name="RecomputedModel",
                streaming_recompute_threshold=16,
            )
            retained = compile_model(
                model, root / "retained", model_name="RetainedModel",
                streaming_recompute_threshold=0,
            )
            self.assertEqual(recomputed["dense_chain_schedule"]["decision_counts"]["recompute"], 1)
            self.assertEqual(recomputed["dense_chain_schedule"]["recompute_extra_madds"], 16)
            self.assertEqual(recomputed["sample_local_storage"]["workspace_elements"], 4)
            self.assertEqual(retained["dense_chain_schedule"]["decision_counts"]["retain"], 1)
            self.assertEqual(retained["sample_local_storage"]["workspace_elements"], 8)
            recomputed_header = (root / "recomputed" / "RecomputedModel.hpp").read_text()
            retained_header = (root / "retained" / "RetainedModel.hpp").read_text()
            streaming_loop = "for (int ihidden = 0; ihidden < 4; ihidden++)"
            self.assertGreater(recomputed_header.count(streaming_loop), retained_header.count(streaming_loop))

    def test_recomputed_branch_extends_materialized_source_liveness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(43)
            widths = ((4, 3), (3, 2), (2, 2), (2, 2))
            initializers = []
            for index, (input_width, output_width) in enumerate(widths):
                initializers.extend([
                    numpy_helper.from_array(
                        rng.standard_normal((input_width, output_width)).astype(np.float32), f"weight{index}"
                    ),
                    numpy_helper.from_array(
                        rng.standard_normal(output_width).astype(np.float32), f"bias{index}"
                    ),
                ])
            nodes = [
                helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
                helper.make_node("Gemm", ["x", "weight0", "bias0"], ["dense0"]),
                helper.make_node("Tanh", ["dense0"], ["base"]),
                helper.make_node("Gemm", ["base", "weight1", "bias1"], ["dense1"]),
                helper.make_node("Tanh", ["dense1"], ["shared"]),
                helper.make_node("Gemm", ["shared", "weight2", "bias2"], ["left"]),
                helper.make_node("Gemm", ["shared", "weight3", "bias3"], ["right"]),
                helper.make_node("Add", ["left", "right"], ["joined"]),
                helper.make_node("Transpose", ["joined"], ["output"], perm=[1, 0]),
            ]
            model = _save_model(
                root / "nested_recompute.onnx", nodes, initializers,
                input_shape=(4, "batch"), output_shape=(2, "batch")
            )
            _, optimized, _ = load_and_optimize(model)
            schedule = schedule_dense_chains(optimized, recompute_madd_threshold=6)
            recompute = next(
                decision for decision in schedule.decisions.values() if decision.action == "recompute"
            )
            source_id = optimized.node_by_id(recompute.producer_id).inputs[0]
            extensions = schedule.recompute_liveness_extensions(optimized)
            self.assertEqual(extensions[source_id], set(recompute.consumer_ids))
            sample_plan = plan_storage(optimized, schedule.eliminated_tensors, extensions)
            source_slot = sample_plan.slots[source_id]
            positions = {node.id: index for index, node in enumerate(optimized.nodes)}
            self.assertEqual(source_slot.last_use, max(positions[node_id] for node_id in recompute.consumer_ids))

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
            _, optimized, _ = load_and_optimize(model, {"elementwise-chain-fusion"})
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
            self.assertIn("outputs(i,ibatch) = inputs(i,ibatch);", generated)

    def test_negative_generation_limits_are_rejected_before_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CompilerError, "--streaming-recompute-threshold must be nonnegative"):
                compile_model(
                    root / "unused.onnx", root / "out", streaming_recompute_threshold=-1
                )

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
            self.assertEqual([node.op for node in optimized.nodes], ["Concat", "Dense"])
            values = np.arange(28, dtype=np.float32).reshape(4, 7) / 11
            np.testing.assert_allclose(run_graph(original, values), run_graph(optimized, values), rtol=1e-6, atol=1e-6)
            compile_model(model, root / "out", model_name="ConcatModel")
            generated = (root / "out" / "ConcatModel.hpp").read_text()
            self.assertIn("workspace[0 + 0 + i] = inputs(i)", generated)
            self.assertIn("ponni::TwoHalf workspace", generated)
            self.assertIn("else if (i < 8)", generated)

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
            _, unfused, _ = load_and_optimize(model, {"elementwise-chain-fusion"})
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
            self.assertEqual([node.op for node in optimized.nodes].count("ElementwiseChain"), 1)
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
            path = Path(directory) / "sin.onnx"
            _save_model(path, [helper.make_node("Sin", ["input"], ["output"])], [], output_shape=(4, "batch"))
            with self.assertRaisesRegex(CompilerError, "unsupported operator 'Sin'"):
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
            blob = bytearray((root / "out" / "weights.bin").read_bytes())
            blob[-1] ^= 0x80
            corrupt = root / "corrupt.bin"
            corrupt.write_bytes(blob)
            with self.assertRaisesRegex(CompilerError, "checksum mismatch"):
                validate_weight_blob(corrupt)


if __name__ == "__main__":
    unittest.main()
