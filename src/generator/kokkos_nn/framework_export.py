from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import CompilerError
from .export import ExportResult


def _dependencies():
    try:
        import keras
        import onnx
        import onnxruntime as ort
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise CompilerError(
            "Keras/TensorFlow export requires keras, tensorflow, tf2onnx, onnx, and onnxruntime"
        ) from exc
    return keras, tf, tf2onnx, onnx, ort


def _set_dimension(dimension, value: int | str) -> None:
    dimension.ClearField("dim_value")
    dimension.ClearField("dim_param")
    if isinstance(value, int):
        dimension.dim_value = value
    else:
        dimension.dim_param = value


def _annotate_feature_batch_model(model_path: Path, num_inputs: int, num_outputs: int, exporter: str) -> None:
    _, _, _, onnx, _ = _dependencies()
    model = onnx.load(model_path)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise CompilerError(
            f"{exporter} produced {len(model.graph.input)} inputs and {len(model.graph.output)} outputs; "
            "PONNI requires exactly one of each"
        )
    boundaries = ((model.graph.input[0], num_inputs), (model.graph.output[0], num_outputs))
    for value, feature_count in boundaries:
        shape = value.type.tensor_type.shape.dim
        if len(shape) != 2:
            raise CompilerError(f"{exporter} produced rank-{len(shape)} boundary tensor {value.name!r}; expected rank 2")
        _set_dimension(shape[0], feature_count)
        _set_dimension(shape[1], "batch")

    metadata = {entry.key: entry for entry in model.metadata_props}
    for key, value in {
        "ponni.orientation": "features_batch",
        "ponni.batch_symbol": "batch",
        "ponni.exporter": exporter,
    }.items():
        if key in metadata:
            metadata[key].value = value
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _write_reference(reference_path: Path, cases: list[tuple[np.ndarray, np.ndarray]]) -> None:
    num_inputs = cases[0][0].shape[0]
    num_outputs = cases[0][1].shape[0]
    with reference_path.open("w") as stream:
        stream.write(f"{len(cases)} {num_inputs} {num_outputs}\n")
        for inputs, outputs in cases:
            stream.write(f"{inputs.shape[1]}\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in inputs.reshape(-1)) + "\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in outputs.reshape(-1)) + "\n")


def _verify_and_write(model_path: Path, reference_path: Path, framework, num_inputs: int,
                      batch_sizes: tuple[int, ...], seed: int) -> ExportResult:
    _, _, _, _, ort = _dependencies()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rng = np.random.default_rng(seed)
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for batch_size in batch_sizes:
        inputs = rng.standard_normal((num_inputs, batch_size)).astype(np.float32)
        expected = np.asarray(framework(inputs), dtype=np.float32)
        actual = session.run([output_name], {input_name: inputs})[0]
        absolute = np.abs(expected - actual)
        relative = absolute / np.maximum(np.abs(expected), 1.e-7)
        maximum_absolute = max(maximum_absolute, float(absolute.max(initial=0.0)))
        maximum_relative = max(maximum_relative, float(relative.max(initial=0.0)))
        if not np.allclose(expected, actual, rtol=2.e-5, atol=2.e-6):
            index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
            raise CompilerError(
                f"ONNX Runtime differs from the source framework at batch {batch_size}, index {index}: "
                f"framework={expected[index]}, onnx={actual[index]}, absolute_error={absolute[index]}"
            )
        cases.append((inputs, expected))
    _write_reference(reference_path, cases)
    return ExportResult(model_path, reference_path, maximum_absolute, maximum_relative)


def export_keras_model(output_dir: str | Path, name: str = "keras_mlp",
                       batch_sizes: tuple[int, ...] = (1, 2, 7, 11), seed: int = 9131) -> ExportResult:
    keras, _, _, _, _ = _dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{name}.onnx"
    reference_path = output_path / f"{name}_reference.txt"

    inputs = keras.Input(batch_shape=(4, None), dtype="float32", name="input")
    values = keras.ops.transpose(inputs, axes=(1, 0))
    values = keras.layers.Dense(6, activation="tanh", name="dense0")(values)
    values = keras.layers.Dense(3, activation="sigmoid", name="dense1")(values)
    outputs = keras.ops.transpose(values, axes=(1, 0))
    model = keras.Model(inputs, outputs, name="ponni_keras_mlp")

    # Build once, then replace backend initialization with deterministic, nonzero parameters.
    model(np.zeros((4, 1), dtype=np.float32), training=False)
    rng = np.random.default_rng(seed)
    for layer in (model.get_layer("dense0"), model.get_layer("dense1")):
        kernel, bias = layer.get_weights()
        layer.set_weights([
            rng.standard_normal(kernel.shape).astype(np.float32) * 0.2,
            rng.standard_normal(bias.shape).astype(np.float32) * 0.1,
        ])
    model.export(model_path, format="onnx", verbose=False)
    _annotate_feature_batch_model(model_path, 4, 3, "keras.Model.export(format='onnx')")

    def evaluate(values: np.ndarray) -> np.ndarray:
        return np.asarray(model(values, training=False), dtype=np.float32)

    return _verify_and_write(model_path, reference_path, evaluate, 4, batch_sizes, seed + 1)


def export_tensorflow_model(output_dir: str | Path, name: str = "tensorflow_residual",
                            batch_sizes: tuple[int, ...] = (1, 2, 7, 11), seed: int = 9173) -> ExportResult:
    _, tf, tf2onnx, _, _ = _dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{name}.onnx"
    reference_path = output_path / f"{name}_reference.txt"
    rng = np.random.default_rng(seed)

    class ResidualModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = tf.Variable(rng.standard_normal((4, 6)).astype(np.float32) * 0.2, trainable=False)
            self.bias0 = tf.Variable(rng.standard_normal(6).astype(np.float32) * 0.1, trainable=False)
            self.weight1 = tf.Variable(rng.standard_normal((6, 4)).astype(np.float32) * 0.2, trainable=False)
            self.bias1 = tf.Variable(rng.standard_normal(4).astype(np.float32) * 0.1, trainable=False)

        @tf.function(input_signature=[tf.TensorSpec([4, None], tf.float32, name="input")])
        @tf.autograph.experimental.do_not_convert
        def __call__(self, values):
            samples = tf.transpose(values, perm=(1, 0))
            hidden = tf.math.tanh(tf.linalg.matmul(samples, self.weight0) + self.bias0)
            branch = tf.linalg.matmul(hidden, self.weight1) + self.bias1
            return tf.transpose(tf.math.sigmoid(branch + samples), perm=(1, 0))

    module = ResidualModule()
    signature = [tf.TensorSpec([4, None], tf.float32, name="input")]
    tf2onnx.convert.from_function(
        module.__call__, input_signature=signature, opset=18, output_path=str(model_path)
    )
    _annotate_feature_batch_model(model_path, 4, 4, "tf2onnx.convert.from_function")

    def evaluate(values: np.ndarray) -> np.ndarray:
        return np.asarray(module(tf.convert_to_tensor(values)).numpy(), dtype=np.float32)

    return _verify_and_write(model_path, reference_path, evaluate, 4, batch_sizes, seed + 1)


def export_keras_normalization_model(output_dir: str | Path, name: str = "keras_normalization",
                                     batch_sizes: tuple[int, ...] = (1, 2, 7, 11),
                                     seed: int = 9199) -> ExportResult:
    keras, _, _, _, _ = _dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{name}.onnx"
    reference_path = output_path / f"{name}_reference.txt"

    inputs = keras.Input(batch_shape=(5, None), dtype="float32", name="input")
    values = keras.ops.transpose(inputs, axes=(1, 0))
    values = keras.layers.Dense(5, name="dense")(values)
    values = keras.layers.BatchNormalization(name="batch_normalization")(values, training=False)
    values = keras.layers.LayerNormalization(axis=-1, name="layer_normalization")(values)
    values = keras.layers.Softmax(axis=-1, name="softmax")(values)
    outputs = keras.ops.transpose(values, axes=(1, 0))
    model = keras.Model(inputs, outputs, name="ponni_keras_normalization")
    model(np.zeros((5, 1), dtype=np.float32), training=False)

    rng = np.random.default_rng(seed)
    dense = model.get_layer("dense")
    kernel, bias = dense.get_weights()
    dense.set_weights([
        rng.standard_normal(kernel.shape).astype(np.float32) * 0.2,
        rng.standard_normal(bias.shape).astype(np.float32) * 0.1,
    ])
    batch_normalization = model.get_layer("batch_normalization")
    gamma, beta, moving_mean, moving_variance = batch_normalization.get_weights()
    batch_normalization.set_weights([
        np.linspace(0.8, 1.2, gamma.size, dtype=np.float32),
        np.linspace(-0.1, 0.1, beta.size, dtype=np.float32),
        np.linspace(-0.2, 0.2, moving_mean.size, dtype=np.float32),
        np.linspace(0.7, 1.3, moving_variance.size, dtype=np.float32),
    ])
    layer_normalization = model.get_layer("layer_normalization")
    layer_gamma, layer_beta = layer_normalization.get_weights()
    layer_normalization.set_weights([
        np.linspace(0.9, 1.1, layer_gamma.size, dtype=np.float32),
        np.linspace(-0.05, 0.05, layer_beta.size, dtype=np.float32),
    ])

    model.export(model_path, format="onnx", verbose=False)
    _annotate_feature_batch_model(model_path, 5, 5, "keras.Model.export(format='onnx')")

    def evaluate(values: np.ndarray) -> np.ndarray:
        return np.asarray(model(values, training=False), dtype=np.float32)

    return _verify_and_write(model_path, reference_path, evaluate, 5, batch_sizes, seed + 1)
