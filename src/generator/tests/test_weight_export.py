"""PONNI-file interoperability and framework-adapter contract tests."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
import warnings

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from onnx import TensorProto, helper
from safetensors import safe_open
from safetensors.numpy import load_file

from kokkos_nn.errors import CompilerError
from kokkos_nn.weight_export import (
    export_jax_flax_weights,
    export_keras_weights,
    export_paddle_weights,
    export_pytorch_weights,
    export_sklearn_weights,
    export_tensorflow_weights,
)
from kokkos_nn.weights import validate_weight_blob, write_ponni_file


class _Variable:
    def __init__(self, name: str, value: np.ndarray):
        self.name = name
        self.path = name
        self._value = value

    def numpy(self) -> np.ndarray:
        return self._value


class _KerasModel:
    def __init__(self):
        self.weights = [_Variable("dense/kernel", np.arange(6, dtype=np.float32).reshape(2, 3))]


class _TensorFlowModule:
    def __init__(self):
        self.variables = [_Variable("dense/bias:0", np.arange(3, dtype=np.float32))]


class _PyTorchModel:
    def state_dict(self):
        return OrderedDict((
            ("dense.weight", np.arange(6, dtype=np.float32).reshape(3, 2)),
            ("dense.bias", np.arange(3, dtype=np.float32)),
        ))


class _SklearnModel:
    coefs_ = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    intercepts_ = [np.arange(3, dtype=np.float32)]


class _PaddleModel:
    def named_parameters(self):
        return [("dense.weight", np.arange(6, dtype=np.float32).reshape(2, 3))]


def _onnx_model(path: Path, operation: str) -> Path:
    node = helper.make_node(operation, ["input"], ["output"], name=f"test_{operation.lower()}")
    graph = helper.make_graph(
        [node], "weight_export_contract",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, "batch"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    for key, value in (("ponni.orientation", "features_batch"), ("ponni.batch_symbol", "batch")):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)
    return path


class WeightExportTests(unittest.TestCase):
    def test_ponni_profile_remains_readable_by_standard_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.ponni"
            expected = np.arange(6, dtype=np.float32).reshape(2, 3)
            write_ponni_file(
                {"dense.kernel": expected, "scalar": np.asarray(2.0, dtype=np.float32)},
                path, source_framework="test",
            )
            manifest = validate_weight_blob(path)

            np.testing.assert_array_equal(load_file(path)["dense.kernel"], expected)
            self.assertEqual(load_file(path)["scalar"].shape, ())
            with safe_open(path, framework="numpy") as handle:
                self.assertEqual(handle.metadata()["ponni.profile_version"], "1")
                self.assertEqual(handle.metadata()["ponni.payload_checksum_fnv1a64"],
                                 manifest["payload_checksum_fnv1a64"])

    def test_payload_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.ponni"
            write_ponni_file({"values": np.arange(4, dtype=np.float32)}, path)
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 0x40
            path.write_bytes(blob)
            with self.assertRaisesRegex(CompilerError, "payload checksum mismatch"):
                validate_weight_blob(path)

    def test_major_framework_adapters_emit_canonical_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = {
                "keras": (export_keras_weights, _KerasModel(), "dense/kernel"),
                "tensorflow": (export_tensorflow_weights, _TensorFlowModule(), "dense/bias"),
                "pytorch": (export_pytorch_weights, _PyTorchModel(), "dense.weight"),
                "jax": (export_jax_flax_weights,
                        {"dense": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}},
                        "dense.kernel"),
                "sklearn": (export_sklearn_weights, _SklearnModel(), "layer.0.weight"),
                "paddle": (export_paddle_weights, _PaddleModel(), "dense.weight"),
            }
            for label, (exporter, model, expected_name) in exports.items():
                with self.subTest(framework=label):
                    path = root / f"{label}.ponni"
                    exporter(model, path, validate_onnx=False)
                    tensors = load_file(path)
                    self.assertIn(expected_name, tensors)
                    validate_weight_blob(path)
            np.testing.assert_array_equal(
                load_file(root / "pytorch.ponni")["dense.weight"],
                np.arange(6, dtype=np.float32).reshape(3, 2).T,
            )

    def test_fitted_sklearn_mlp_exports_exact_predictive_parameters(self) -> None:
        """Exercise fitted framework state rather than a coefs_/intercepts_ stand-in."""
        inputs = np.asarray([
            [-1.0, -0.5], [-0.5, 0.25], [0.0, -1.0], [0.25, 0.75],
            [0.5, -0.25], [0.75, 1.0], [1.0, 0.5], [1.5, -0.75],
        ], dtype=np.float64)
        targets = np.sin(inputs[:, 0]) + 0.25 * inputs[:, 1]
        model = MLPRegressor(
            hidden_layer_sizes=(3,), activation="tanh", solver="lbfgs",
            max_iter=500, random_state=1729, tol=1.e-10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(inputs, targets)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sklearn.ponni"
            export_sklearn_weights(model, path, validate_onnx=False)
            tensors = load_file(path)
            validate_weight_blob(path)

            for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_, strict=True)):
                np.testing.assert_array_equal(tensors[f"layer.{index}.weight"], weights)
                np.testing.assert_array_equal(tensors[f"layer.{index}.bias"], bias)

            # Re-evaluate the fitted MLP solely from exported arrays. This checks
            # both layer ordering and the input/output orientation expected by PONNI.
            hidden = np.tanh(inputs @ tensors["layer.0.weight"] + tensors["layer.0.bias"])
            exported_prediction = hidden @ tensors["layer.1.weight"] + tensors["layer.1.bias"]
            np.testing.assert_allclose(exported_prediction[:, 0], model.predict(inputs), rtol=1.e-12, atol=1.e-12)

    def test_trained_flax_model_exports_real_jax_parameter_tree(self) -> None:
        """Initialize a Flax module and update its genuine JAX arrays before export."""
        class TinyFlaxMlp(nn.Module):
            @nn.compact
            def __call__(self, values):
                values = nn.Dense(3, name="hidden")(values)
                values = nn.tanh(values)
                return nn.Dense(1, name="output")(values)

        model = TinyFlaxMlp()
        inputs = jnp.asarray([
            [-1.0, 0.5], [-0.25, -0.5], [0.5, 0.25], [1.0, -0.75],
        ], dtype=jnp.float32)
        targets = jnp.asarray([[-0.75], [-0.5], [0.75], [0.25]], dtype=jnp.float32)
        parameters = model.init(jax.random.key(811), inputs)["params"]

        def loss(candidate):
            residual = model.apply({"params": candidate}, inputs) - targets
            return jnp.mean(residual * residual)

        gradients = jax.grad(loss)(parameters)
        trained = jax.tree_util.tree_map(lambda value, gradient: value - 0.05 * gradient,
                                         parameters, gradients)
        self.assertTrue(all(isinstance(value, jax.Array) for value in jax.tree_util.tree_leaves(trained)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flax.ponni"
            export_jax_flax_weights(trained, path, validate_onnx=False)
            tensors = load_file(path)
            validate_weight_blob(path)

            expected_names = {"hidden.bias", "hidden.kernel", "output.bias", "output.kernel"}
            self.assertEqual(set(tensors), expected_names)
            for layer in ("hidden", "output"):
                for parameter in ("bias", "kernel"):
                    np.testing.assert_array_equal(
                        tensors[f"{layer}.{parameter}"], np.asarray(trained[layer][parameter])
                    )

            hidden = np.tanh(np.asarray(inputs) @ tensors["hidden.kernel"] + tensors["hidden.bias"])
            exported_prediction = hidden @ tensors["output.kernel"] + tensors["output.bias"]
            framework_prediction = np.asarray(model.apply({"params": trained}, inputs))
            np.testing.assert_allclose(exported_prediction, framework_prediction, rtol=2.e-6, atol=2.e-6)

    def test_generator_onnx_support_is_validated_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supported = _onnx_model(root / "supported.onnx", "Relu")
            result = export_jax_flax_weights(
                {"values": np.arange(2, dtype=np.float32)}, root / "supported.ponni",
                onnx_path=supported,
            )
            self.assertIn("onnx_validation", result)

            unsupported = _onnx_model(root / "unsupported.onnx", "ArgMax")
            with self.assertRaises(CompilerError) as context:
                export_jax_flax_weights(
                    {"values": np.arange(2, dtype=np.float32)}, root / "unsupported.ponni",
                    onnx_path=unsupported,
                )
            diagnostic = str(context.exception)
            self.assertIn("jax-flax ONNX model is not supported", diagnostic)
            self.assertIn("Operator inventory", diagnostic)
            self.assertIn("ai.onnx::ArgMax", diagnostic)
            self.assertIn("test_argmax", diagnostic)
            self.assertIn("Boundaries", diagnostic)
            self.assertIn("PONNI diagnostic", diagnostic)


if __name__ == "__main__":
    unittest.main()
