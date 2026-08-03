from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CompilerError


@dataclass
class ExportResult:
    model_path: Path
    reference_path: Path
    max_onnx_absolute_error: float
    max_onnx_relative_error: float


def _dependencies():
    try:
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise CompilerError("PyTorch export requires torch, onnx, onnxscript, and onnxruntime") from exc
    return torch, onnx, ort


def export_module(module, num_inputs: int, output_dir: str | Path, name: str,
                  batch_sizes: tuple[int, ...] = (1, 2, 7, 32, 67), seed: int = 8128) -> ExportResult:
    torch, onnx, ort = _dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{name}.onnx"
    reference_path = output_path / f"{name}_reference.txt"

    class FeatureBatchWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, features_batch):
            return self.wrapped(features_batch.transpose(0, 1)).transpose(0, 1)

    wrapped = FeatureBatchWrapper(module.eval()).eval()
    generator = torch.Generator().manual_seed(seed)
    example = torch.randn((num_inputs, 7), generator=generator, dtype=torch.float32)
    batch_dimension = torch.export.Dim("batch", min=1, max=4096)
    torch.onnx.export(
        wrapped,
        args=(example,),
        f=model_path,
        dynamo=True,
        optimize=True,
        verify=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({1: batch_dimension},),
    )

    model = onnx.load(model_path)
    metadata = {entry.key: entry for entry in model.metadata_props}
    for key, value in {
        "ponni.orientation": "features_batch",
        "ponni.batch_symbol": "batch",
        "ponni.exporter": "torch.onnx.export(dynamo=True,optimize=True,verify=True)",
    }.items():
        if key in metadata:
            metadata[key].value = value
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
    onnx.save(model, model_path)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for batch_size in batch_sizes:
        values = torch.randn((num_inputs, batch_size), generator=generator, dtype=torch.float32)
        with torch.no_grad():
            torch_output = wrapped(values).cpu().numpy()
        onnx_output = session.run(["output"], {"input": values.cpu().numpy()})[0]
        absolute = np.abs(torch_output - onnx_output)
        relative = absolute / np.maximum(np.abs(torch_output), 1e-7)
        maximum_absolute = max(maximum_absolute, float(absolute.max(initial=0.0)))
        maximum_relative = max(maximum_relative, float(relative.max(initial=0.0)))
        if not np.allclose(torch_output, onnx_output, rtol=2e-5, atol=2e-6):
            index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
            raise CompilerError(
                f"ONNX Runtime differs from PyTorch for {name}, batch {batch_size}, index {index}: "
                f"torch={torch_output[index]}, onnx={onnx_output[index]}, absolute_error={absolute[index]}"
            )
        cases.append((values.cpu().numpy(), torch_output))

    with reference_path.open("w") as stream:
        stream.write(f"{len(cases)} {num_inputs} {cases[0][1].shape[0]}\n")
        for inputs, outputs in cases:
            stream.write(f"{inputs.shape[1]}\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in inputs.reshape(-1)) + "\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in outputs.reshape(-1)) + "\n")

    return ExportResult(model_path, reference_path, maximum_absolute, maximum_relative)


def make_example_models():
    torch, _, _ = _dependencies()
    torch.manual_seed(20260802)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense0 = torch.nn.Linear(4, 5)
            self.dense1 = torch.nn.Linear(5, 3)

        def forward(self, value):
            return self.dense1(torch.tanh(self.dense0(value)))

    class ResidualMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense0 = torch.nn.Linear(4, 4)
            self.dense1 = torch.nn.Linear(4, 4)

        def forward(self, value):
            branch = self.dense1(torch.tanh(self.dense0(value)))
            return torch.sigmoid(branch + value)

    return MLP().eval(), ResidualMLP().eval()


def make_functionality_models():
    """Return small deterministic DAGs that stress depth, liveness, branching, and activation diversity."""
    torch, _, _ = _dependencies()
    torch.manual_seed(20260803)

    def activate(value, kind: int):
        if kind % 3 == 0:
            return torch.relu(value)
        if kind % 3 == 1:
            return torch.tanh(value)
        return torch.sigmoid(value)

    class DeepTen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dimensions = (9, 11, 8, 12, 10, 7, 13, 9, 11, 8, 7)
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(dimensions[index], dimensions[index + 1])
                for index in range(len(dimensions) - 1)
            )

        def forward(self, value):
            for index, layer in enumerate(self.layers):
                value = layer(value)
                if index + 1 < len(self.layers):
                    value = activate(value, index)
            return value

    class ResNetTen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Linear(10, 10) for _ in range(10))

        def forward(self, value):
            for block in range(5):
                hidden = activate(self.layers[2 * block](value), block)
                value = activate(self.layers[2 * block + 1](hidden) + value, block + 1)
            return value

    class DenseNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(8, 2), torch.nn.Linear(10, 2),
                 torch.nn.Linear(12, 2), torch.nn.Linear(14, 2)]
            )
            self.output = torch.nn.Linear(16, 9)

        def forward(self, value):
            features = [value]
            for index, block in enumerate(self.blocks):
                joined = torch.cat(features, dim=1)
                features.append(activate(block(joined), index))
            return self.output(torch.cat(features, dim=1))

    class Branching(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(11, 4)
            self.left = torch.nn.Linear(4, 4)
            self.right = torch.nn.Linear(4, 4)
            self.output = torch.nn.Linear(4, 8)

        def forward(self, value):
            shared = torch.tanh(self.trunk(value))
            left = torch.relu(self.left(shared))
            right = torch.sigmoid(self.right(shared))
            return self.output(torch.tanh(left + right))

    return {
        "deep10": (DeepTen().eval(), 9),
        "resnet10": (ResNetTen().eval(), 10),
        "densenet": (DenseNet().eval(), 8),
        "branching": (Branching().eval(), 11),
    }


def export_operator_zoo(output_dir: str | Path, batch_sizes: tuple[int, ...] = (1, 2, 3, 7, 11),
                        seed: int = 20260804) -> ExportResult:
    """Create a deterministic ONNX fixture covering the broader scalar/reduction operator families."""
    _, onnx, ort = _dependencies()
    from onnx import TensorProto, helper, numpy_helper

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "operator_zoo.onnx"
    reference_path = output_path / "operator_zoo_reference.txt"
    width = 8
    rng = np.random.default_rng(seed)
    arrays = {
        "bn_scale": rng.uniform(0.7, 1.3, width).astype(np.float32),
        "bn_bias": rng.uniform(-0.2, 0.2, width).astype(np.float32),
        "bn_mean": rng.uniform(-0.3, 0.3, width).astype(np.float32),
        "bn_var": rng.uniform(0.7, 1.4, width).astype(np.float32),
        "ln_scale": rng.uniform(0.8, 1.2, width).astype(np.float32),
        "ln_bias": rng.uniform(-0.1, 0.1, width).astype(np.float32),
        "one": np.array(1.25, dtype=np.float32),
        "clip_min": np.array(-1.0, dtype=np.float32),
        "clip_max": np.array(1.0, dtype=np.float32),
        "two": np.array(2.0, dtype=np.float32),
        "upper": np.array(0.75, dtype=np.float32),
        "lower": np.array(0.05, dtype=np.float32),
        "axes": np.array([1], dtype=np.int64),
    }
    nodes = [
        helper.make_node("Transpose", ["input"], ["x"], perm=[1, 0]),
        helper.make_node("BatchNormalization", ["x", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["bn"]),
        helper.make_node("LeakyRelu", ["bn"], ["leaky"], alpha=0.125),
        helper.make_node("Elu", ["leaky"], ["elu"], alpha=0.75),
        helper.make_node("Gelu", ["elu"], ["gelu"], approximate="tanh"),
        helper.make_node("Softplus", ["gelu"], ["softplus"]),
        helper.make_node("HardSigmoid", ["softplus"], ["hard_sigmoid"], alpha=0.2, beta=0.5),
        helper.make_node("HardSwish", ["hard_sigmoid"], ["hard_swish"]),
        helper.make_node("Mish", ["hard_swish"], ["mish"]),
        helper.make_node("Sigmoid", ["mish"], ["silu_gate"]),
        helper.make_node("Mul", ["mish", "silu_gate"], ["silu"]),
        helper.make_node("Abs", ["silu"], ["absolute"]),
        helper.make_node("Add", ["absolute", "one"], ["positive"]),
        helper.make_node("Sqrt", ["positive"], ["root"]),
        helper.make_node("Log", ["root"], ["logged"]),
        helper.make_node("Exp", ["logged"], ["exponential"]),
        helper.make_node("Neg", ["exponential"], ["negative"]),
        helper.make_node("Clip", ["negative", "clip_min", "clip_max"], ["clipped"]),
        helper.make_node("Pow", ["clipped", "two"], ["powered"]),
        helper.make_node("Min", ["powered", "upper"], ["minimum"]),
        helper.make_node("Max", ["minimum", "lower"], ["maximum"]),
        helper.make_node("LayerNormalization", ["maximum", "ln_scale", "ln_bias"], ["normalized"], axis=1),
        helper.make_node("Softmax", ["normalized"], ["probability"], axis=1),
        helper.make_node("LogSoftmax", ["normalized"], ["log_probability"], axis=1),
        helper.make_node("ReduceMean", ["log_probability", "axes"], ["log_mean"], keepdims=1),
        helper.make_node("ReduceSum", ["probability", "axes"], ["probability_sum"], keepdims=1),
        helper.make_node("Add", ["probability", "log_probability"], ["combined"]),
        helper.make_node("Add", ["combined", "log_mean"], ["centered"]),
        helper.make_node("Add", ["centered", "probability_sum"], ["result"]),
        helper.make_node("Transpose", ["result"], ["output"], perm=[1, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        "ponni_operator_zoo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [width, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [width, "batch"])],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    for key, value in {"ponni.orientation": "features_batch", "ponni.batch_symbol": "batch"}.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_size in batch_sizes:
        inputs = rng.standard_normal((width, batch_size)).astype(np.float32)
        outputs = session.run(["output"], {"input": inputs})[0]
        cases.append((inputs, outputs))
    with reference_path.open("w") as stream:
        stream.write(f"{len(cases)} {width} {width}\n")
        for inputs, outputs in cases:
            stream.write(f"{inputs.shape[1]}\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in inputs.reshape(-1)) + "\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in outputs.reshape(-1)) + "\n")
    return ExportResult(model_path, reference_path, 0.0, 0.0)
