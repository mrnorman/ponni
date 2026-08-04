from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CompilerError
from .onnx_reference import run_onnx_reference


@dataclass
class ExportResult:
    model_path: Path
    reference_path: Path
    max_onnx_absolute_error: float
    max_onnx_relative_error: float


def _dependencies():
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise CompilerError("PyTorch export requires torch, onnx, onnxscript, and CPU onnxruntime") from exc
    return torch, onnx


def _set_dimension(dimension, value: int | str) -> None:
    dimension.ClearField("dim_value")
    dimension.ClearField("dim_param")
    if isinstance(value, int):
        dimension.dim_value = value
    else:
        dimension.dim_param = value


def _normalize_feature_batch_boundaries(model, num_inputs: int, exporter: str) -> None:
    """Make exporter-chosen symbolic names irrelevant to the PONNI boundary contract."""
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise CompilerError(
            f"{exporter} produced {len(model.graph.input)} inputs and {len(model.graph.output)} outputs; "
            "PONNI requires exactly one of each"
        )
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise CompilerError(
            f"{exporter} produced boundary ranks {len(input_shape)} and {len(output_shape)}; expected rank 2"
        )
    if output_shape[0].dim_value <= 0:
        raise CompilerError(f"{exporter} did not infer a static output feature count")
    _set_dimension(input_shape[0], num_inputs)
    _set_dimension(input_shape[1], "batch")
    _set_dimension(output_shape[1], "batch")


def export_module(module, num_inputs: int, output_dir: str | Path, name: str,
                  batch_sizes: tuple[int, ...] = (1, 2, 7, 32, 67), seed: int = 8128) -> ExportResult:
    torch, onnx = _dependencies()
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
    _normalize_feature_batch_boundaries(model, num_inputs, "torch.onnx.export")
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

    cases: list[tuple[np.ndarray, np.ndarray]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for batch_size in batch_sizes:
        values = torch.randn((num_inputs, batch_size), generator=generator, dtype=torch.float32)
        with torch.no_grad():
            torch_output = wrapped(values).cpu().numpy()
        onnx_output = run_onnx_reference(model_path, ["output"], {"input": values.cpu().numpy()})[0]
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
    torch, _ = _dependencies()
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
    torch, _ = _dependencies()
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
    _, onnx = _dependencies()
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
        "zero": np.array(0.0, dtype=np.float32),
        "clip_min": np.array(-1.0, dtype=np.float32),
        "clip_max": np.array(1.0, dtype=np.float32),
        "two": np.array(2.0, dtype=np.float32),
        "upper": np.array(0.75, dtype=np.float32),
        "lower": np.array(0.05, dtype=np.float32),
        "axes": np.array([1], dtype=np.int64),
        "feature_indices": np.arange(width - 1, -1, -1, dtype=np.int64),
        "singleton_axis": np.array([1], dtype=np.int64),
        "shape_index": np.array(0, dtype=np.int64),
        "prelu_slope": np.array(0.25, dtype=np.float32),
    }
    nodes = [
        helper.make_node("Transpose", ["input"], ["raw_x"], perm=[1, 0]),
        helper.make_node("Dropout", ["raw_x"], ["dropout_x"]),
        helper.make_node("Unsqueeze", ["dropout_x", "singleton_axis"], ["expanded_x"]),
        helper.make_node("Squeeze", ["expanded_x", "singleton_axis"], ["squeezed_x"]),
        helper.make_node("CastLike", ["squeezed_x", "one"], ["cast_like_x"]),
        helper.make_node("Gather", ["cast_like_x", "feature_indices"], ["x"], axis=1),
        helper.make_node("Shape", ["bn_scale"], ["static_shape"]),
        helper.make_node("Size", ["bn_scale"], ["static_size"]),
        helper.make_node("Gather", ["static_shape", "shape_index"], ["static_dimension"], axis=0),
        helper.make_node("Greater", ["x", "zero"], ["greater"]),
        helper.make_node("GreaterOrEqual", ["x", "zero"], ["greater_equal"]),
        helper.make_node("Less", ["x", "one"], ["less"]),
        helper.make_node("LessOrEqual", ["x", "one"], ["less_equal"]),
        helper.make_node("Equal", ["greater", "greater_equal"], ["equal"]),
        helper.make_node("And", ["greater", "less_equal"], ["logical_and"]),
        helper.make_node("Not", ["logical_and"], ["logical_not"]),
        helper.make_node("Or", ["less", "equal"], ["logical_or"]),
        helper.make_node("Xor", ["logical_or", "logical_not"], ["logical_xor"]),
        helper.make_node("Cast", ["logical_xor"], ["logical_value"], to=TensorProto.FLOAT),
        helper.make_node("Neg", ["x"], ["negative_x"]),
        helper.make_node("Where", ["logical_xor", "x", "negative_x"], ["selected"]),
        helper.make_node("Add", ["selected", "logical_value"], ["selected_with_mask"]),
        helper.make_node("Greater", ["selected_with_mask", "zero"], ["direct_condition"]),
        helper.make_node("Neg", ["selected_with_mask"], ["negative_selected"]),
        helper.make_node(
            "Where", ["direct_condition", "selected_with_mask", "negative_selected"], ["direct_selected"]
        ),
        helper.make_node(
            "BatchNormalization", ["direct_selected", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["bn"]
        ),
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
        helper.make_node("Sin", ["maximum"], ["sine"]),
        helper.make_node("Cos", ["sine"], ["cosine"]),
        helper.make_node("Tan", ["cosine"], ["tangent"]),
        helper.make_node("Atan", ["tangent"], ["arctangent"]),
        helper.make_node("Acos", ["arctangent"], ["arccosine"]),
        helper.make_node("Asin", ["arccosine"], ["arcsine"]),
        helper.make_node("Atanh", ["arcsine"], ["inverse_hyperbolic_tangent"]),
        helper.make_node("Asinh", ["inverse_hyperbolic_tangent"], ["inverse_hyperbolic_sine"]),
        helper.make_node("Sinh", ["inverse_hyperbolic_sine"], ["hyperbolic_sine"]),
        helper.make_node("Cosh", ["hyperbolic_sine"], ["hyperbolic_cosine"]),
        helper.make_node("Acosh", ["hyperbolic_cosine"], ["inverse_hyperbolic_cosine"]),
        helper.make_node("Erf", ["inverse_hyperbolic_cosine"], ["error_function"]),
        helper.make_node("Ceil", ["error_function"], ["ceiling"]),
        helper.make_node("Floor", ["ceiling"], ["floor"]),
        helper.make_node("Round", ["floor"], ["rounded"]),
        helper.make_node("Sign", ["rounded"], ["signed"]),
        helper.make_node("Add", ["maximum", "signed"], ["math_result"]),
        helper.make_node("LayerNormalization", ["math_result", "ln_scale", "ln_bias"], ["normalized"], axis=1),
        helper.make_node("Celu", ["normalized"], ["celu"], alpha=0.75),
        helper.make_node("Selu", ["celu"], ["selu"]),
        helper.make_node("Softsign", ["selu"], ["softsign"]),
        helper.make_node("ThresholdedRelu", ["softsign"], ["thresholded"], alpha=0.05),
        helper.make_node("PRelu", ["thresholded", "prelu_slope"], ["prelu"]),
        helper.make_node("IsNaN", ["prelu"], ["is_nan"]),
        helper.make_node("IsInf", ["prelu"], ["is_inf"]),
        helper.make_node("Or", ["is_nan", "is_inf"], ["non_finite"]),
        helper.make_node("Cast", ["non_finite"], ["non_finite_value"], to=TensorProto.FLOAT),
        helper.make_node("Mean", ["prelu", "normalized"], ["extended_mean"]),
        helper.make_node("Sum", ["extended_mean", "non_finite_value"], ["extended"]),
        helper.make_node("LpNormalization", ["extended"], ["lp_normalized"], axis=1, p=2),
        helper.make_node("Abs", ["extended"], ["reduction_abs"]),
        helper.make_node("Add", ["reduction_abs", "one"], ["reduction_positive"]),
        helper.make_node("ReduceL1", ["reduction_positive", "axes"], ["reduce_l1_flat"], keepdims=0),
        helper.make_node("Unsqueeze", ["reduce_l1_flat", "singleton_axis"], ["reduce_l1"]),
        helper.make_node("ReduceL2", ["reduction_positive", "axes"], ["reduce_l2"], keepdims=1),
        helper.make_node("ReduceLogSum", ["reduction_positive", "axes"], ["reduce_log_sum"], keepdims=1),
        helper.make_node("ReduceLogSumExp", ["extended", "axes"], ["reduce_log_sum_exp"], keepdims=1),
        helper.make_node("ReduceMax", ["extended", "axes"], ["reduce_max"], keepdims=1),
        helper.make_node("ReduceMin", ["extended", "axes"], ["reduce_min"], keepdims=1),
        helper.make_node("ReduceProd", ["reduction_positive", "axes"], ["reduce_prod"], keepdims=1),
        helper.make_node("ReduceSumSquare", ["extended", "axes"], ["reduce_sum_square"], keepdims=1),
        helper.make_node(
            "Sum",
            ["reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_max", "reduce_min",
             "reduce_prod", "reduce_sum_square"],
            ["reduction_summary"],
        ),
        helper.make_node("Softmax", ["lp_normalized"], ["probability"], axis=1),
        helper.make_node("LogSoftmax", ["lp_normalized"], ["log_probability"], axis=1),
        helper.make_node("ReduceMean", ["log_probability", "axes"], ["log_mean"], keepdims=1),
        helper.make_node("ReduceSum", ["probability", "axes"], ["probability_sum"], keepdims=1),
        helper.make_node("Add", ["probability", "log_probability"], ["combined"]),
        helper.make_node("Add", ["combined", "log_mean"], ["centered"]),
        helper.make_node("Add", ["centered", "probability_sum"], ["result"]),
        helper.make_node("Add", ["result", "reduction_summary"], ["result_with_reductions"]),
        helper.make_node("Transpose", ["result_with_reductions"], ["output"], perm=[1, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        "ponni_operator_zoo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [width, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [width, "batch"])],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()],
    )
    # ONNX Runtime 1.23 does not provide a CPU implementation for the
    # LpNormalization schema introduced in opset 22.  Opset 21 selects the
    # equivalent version-1 schema while retaining the opset-20 operators in
    # this fixture (notably Gelu), so the reference model remains portable.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    for key, value in {"ponni.orientation": "features_batch", "ponni.batch_symbol": "batch"}.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    cases: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_size in batch_sizes:
        inputs = rng.standard_normal((width, batch_size)).astype(np.float32)
        outputs = run_onnx_reference(model_path, ["output"], {"input": inputs})[0]
        cases.append((inputs, outputs))
    with reference_path.open("w") as stream:
        stream.write(f"{len(cases)} {width} {width}\n")
        for inputs, outputs in cases:
            stream.write(f"{inputs.shape[1]}\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in inputs.reshape(-1)) + "\n")
            stream.write(" ".join(f"{float(value):.9g}" for value in outputs.reshape(-1)) + "\n")
    return ExportResult(model_path, reference_path, 0.0, 0.0)
