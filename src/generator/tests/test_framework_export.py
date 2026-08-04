from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import onnx
import numpy as np
import keras
import tensorflow as tf
import tf2onnx

from kokkos_nn.compiler import compile_model
from kokkos_nn.errors import CompilerError
from kokkos_nn.framework_export import (
    _annotate_feature_batch_model,
    export_keras_model,
    export_keras_normalization_model,
    export_tensorflow_model,
)
from kokkos_nn.importer import import_onnx
from kokkos_nn.onnx_reference import run_onnx_reference


class FrameworkExportTests(unittest.TestCase):
    def _assert_onnx_matches(self, model_path: Path, inputs: np.ndarray, expected: np.ndarray) -> None:
        model = onnx.load(model_path)
        input_name = model.graph.input[0].name
        output_name = model.graph.output[0].name
        actual = run_onnx_reference(
            model_path, [output_name], {input_name: inputs.astype(np.float32)}
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=2.e-5, atol=2.e-6)

    def _check_export(self, model_path: Path, output_dir: Path, model_name: str, exporter_prefix: str) -> dict:
        model = onnx.load(model_path)
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        self.assertEqual(metadata["ponni.orientation"], "features_batch")
        self.assertEqual(metadata["ponni.batch_symbol"], "batch")
        self.assertTrue(metadata["ponni.exporter"].startswith(exporter_prefix))
        graph = import_onnx(model_path)
        self.assertGreaterEqual(len(graph.nodes), 5)
        report = compile_model(model_path, output_dir, model_name=model_name)
        self.assertEqual(report["storage"]["external_workspace_bytes"], 0)
        self.assertLessEqual(report["ir_optimization_max_absolute_error"], 2.e-6)
        self.assertTrue((output_dir / f"{model_name}.hpp").is_file())
        self.assertTrue((output_dir / "weights.bin").is_file())
        return report

    def test_keras_and_tensorflow_models_export_import_fuse_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keras_result = export_keras_model(root, batch_sizes=(1, 3, 7))
            normalization_result = export_keras_normalization_model(root, batch_sizes=(1, 3, 7))
            tensorflow_result = export_tensorflow_model(root, batch_sizes=(1, 3, 7))
            self.assertLessEqual(keras_result.max_onnx_absolute_error, 2.e-6)
            self.assertLessEqual(normalization_result.max_onnx_absolute_error, 2.e-6)
            self.assertLessEqual(tensorflow_result.max_onnx_absolute_error, 2.e-6)

            keras_report = self._check_export(
                keras_result.model_path, root / "keras_generated", "KerasModel", "keras.Model.export"
            )
            tensorflow_report = self._check_export(
                tensorflow_result.model_path,
                root / "tensorflow_generated",
                "TensorFlowModel",
                "tf2onnx.convert.from_function",
            )
            normalization_report = self._check_export(
                normalization_result.model_path,
                root / "normalization_generated",
                "KerasNormalizationModel",
                "keras.Model.export",
            )
            self.assertEqual(keras_report["optimized_operations"].count("DenseBiasActivation"), 2)
            self.assertIn("ResidualAddActivation", tensorflow_report["optimized_operations"])
            self.assertIn("Reciprocal", normalization_report["optimized_operations"])
            self.assertIn("Softmax", normalization_report["optimized_operations"])

    def test_keras_functional_branching_no_bias_concat_and_residual_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = keras.Input(batch_shape=(5, None), dtype="float32", name="input")
            samples = keras.ops.transpose(inputs, axes=(1, 0))
            shared = keras.layers.Dense(4, use_bias=False, activation="relu", name="shared")(samples)
            left = keras.layers.Dense(2, activation="tanh", bias_initializer="ones", name="left")(shared)
            right = keras.layers.Dense(
                2, activation="sigmoid", bias_initializer=keras.initializers.Constant(0.5), name="right"
            )(shared)
            joined = keras.layers.Concatenate(axis=1)([left, right])
            projected = keras.layers.Dense(
                5, bias_initializer=keras.initializers.Constant(0.25), name="projection"
            )(joined)
            outputs = keras.ops.transpose(keras.layers.ReLU()(projected + samples), axes=(1, 0))
            model = keras.Model(inputs, outputs)
            values = np.random.default_rng(31).standard_normal((5, 7)).astype(np.float32)
            expected = np.asarray(model(values, training=False), dtype=np.float32)
            model_path = root / "keras_branch.onnx"
            model.export(model_path, format="onnx", verbose=False)
            _annotate_feature_batch_model(model_path, 5, 5, "keras.Model.export representation test")
            self._assert_onnx_matches(model_path, values, expected)

            report = compile_model(model_path, root / "generated", model_name="KerasBranchModel")
            self.assertIn("Concat", report["optimized_operations"])
            self.assertIn("ResidualAddActivation", report["optimized_operations"])
            self.assertEqual(report["learned_parameter_count"], 65)
            self.assertGreaterEqual(report["dense_chain_schedule"]["decision_counts"]["retain"], 1)

    def test_keras_activation_layers_keep_supported_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = keras.Input(batch_shape=(4, None), dtype="float32", name="input")
            values = keras.ops.transpose(inputs, axes=(1, 0))
            values = keras.layers.LeakyReLU(negative_slope=0.2)(values)
            values = keras.layers.Activation("softplus")(values)
            outputs = keras.ops.transpose(values, axes=(1, 0))
            model = keras.Model(inputs, outputs)
            test_values = np.random.default_rng(43).standard_normal((4, 7)).astype(np.float32)
            expected = np.asarray(model(test_values, training=False), dtype=np.float32)
            model_path = root / "keras_activations.onnx"
            model.export(model_path, format="onnx", verbose=False)
            _annotate_feature_batch_model(model_path, 4, 4, "keras activation representation test")
            self._assert_onnx_matches(model_path, test_values, expected)

            graph = import_onnx(model_path)
            operations = [node.op for node in graph.nodes]
            self.assertEqual(operations.count("LeakyRelu"), 1)
            self.assertEqual(operations.count("Softplus"), 1)
            leaky = next(node for node in graph.nodes if node.op == "LeakyRelu")
            self.assertAlmostEqual(float(leaky.attributes["alpha"]), 0.2, places=6)
            report = compile_model(model_path, root / "generated", model_name="KerasActivationModel")
            self.assertEqual(report["optimized_operations"], ["LeakyRelu", "Softplus"])

    def test_keras_elu_boolean_select_decomposition_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = keras.Input(batch_shape=(4, None), dtype="float32", name="input")
            values = keras.ops.transpose(inputs, axes=(1, 0))
            values = keras.layers.ELU(alpha=0.7)(values)
            outputs = keras.ops.transpose(values, axes=(1, 0))
            model = keras.Model(inputs, outputs)
            model(np.zeros((4, 1), dtype=np.float32))
            test_values = np.random.default_rng(47).standard_normal((4, 9)).astype(np.float32)
            expected = np.asarray(model(test_values, training=False), dtype=np.float32)
            model_path = root / "keras_elu.onnx"
            model.export(model_path, format="onnx", verbose=False)
            _annotate_feature_batch_model(model_path, 4, 4, "keras ELU representation test")
            self._assert_onnx_matches(model_path, test_values, expected)
            operations = [node.op_type for node in onnx.load(model_path).graph.node]
            self.assertIn("Elu", operations)
            self.assertIn("Greater", operations)
            self.assertIn("Cast", operations)
            report = compile_model(model_path, root / "generated", model_name="KerasEluModel")
            self.assertGreater(report["sample_local_storage"]["mask_workspace_elements"], 0)
            self.assertIn("Cast", report["optimized_operations"])

    def test_tensorflow_transposed_weight_bias_add_reshape_and_shared_branches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(37)

            class RepresentationModule(tf.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = tf.Variable(rng.standard_normal((4, 5)).astype(np.float32) * 0.2, trainable=False)
                    self.bias = tf.Variable(rng.standard_normal(4).astype(np.float32) * 0.1, trainable=False)
                    self.left = tf.Variable(rng.standard_normal((4, 2)).astype(np.float32) * 0.2, trainable=False)
                    self.right = tf.Variable(rng.standard_normal((4, 2)).astype(np.float32) * 0.2, trainable=False)

                @tf.function(input_signature=[tf.TensorSpec([5, None], tf.float32, name="input")])
                @tf.autograph.experimental.do_not_convert
                def __call__(self, values):
                    samples = tf.transpose(values, perm=(1, 0))
                    shared = tf.nn.relu(tf.nn.bias_add(tf.matmul(samples, self.weight, transpose_b=True), self.bias))
                    shared = tf.reshape(shared, (-1, 4))
                    left = tf.math.tanh(tf.matmul(shared, self.left))
                    right = tf.math.sigmoid(tf.matmul(shared, self.right))
                    return tf.transpose(tf.concat((left, right), axis=1), perm=(1, 0))

            module = RepresentationModule()
            signature = [tf.TensorSpec([5, None], tf.float32, name="input")]
            model_path = root / "tensorflow_representation.onnx"
            tf2onnx.convert.from_function(
                module.__call__, input_signature=signature, opset=18, output_path=str(model_path)
            )
            _annotate_feature_batch_model(model_path, 5, 4, "tf2onnx representation test")
            values = rng.standard_normal((5, 7)).astype(np.float32)
            expected = module(tf.convert_to_tensor(values)).numpy()
            self._assert_onnx_matches(model_path, values, expected)

            onnx_model = onnx.load(model_path)
            self.assertIn("MatMul", [node.op_type for node in onnx_model.graph.node])
            report = compile_model(model_path, root / "generated", model_name="TensorFlowRepresentationModel")
            self.assertEqual(report["optimized_operations"].count("DenseBiasActivation"), 3)
            self.assertIn("Concat", report["optimized_operations"])
            self.assertGreaterEqual(report["dense_chain_schedule"]["decision_counts"]["retain"], 1)

    def test_tensorflow_exported_unsupported_operation_has_actionable_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            @tf.function(input_signature=[tf.TensorSpec([4, None], tf.float32, name="input")])
            @tf.autograph.experimental.do_not_convert
            def unsupported(values):
                return tf.expand_dims(tf.argmax(values, axis=0, output_type=tf.int64), axis=0)

            model_path = root / "tensorflow_argmax.onnx"
            signature = [tf.TensorSpec([4, None], tf.float32, name="input")]
            tf2onnx.convert.from_function(unsupported, input_signature=signature, opset=18, output_path=str(model_path))
            _annotate_feature_batch_model(model_path, 4, 1, "tf2onnx unsupported-operation test")
            with self.assertRaisesRegex(CompilerError, "unsupported operator 'ArgMax'"):
                compile_model(model_path, root / "generated", model_name="UnsupportedModel")


if __name__ == "__main__":
    unittest.main()
