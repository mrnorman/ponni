from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CompilerError
from .ir import ConstantTensor, DType, Graph, Node, Symbol, TensorValue


MIN_SUPPORTED_IR_VERSION = 8
MAX_SUPPORTED_IR_VERSION = 13
MIN_SUPPORTED_ONNX_OPSET = 13
MAX_SUPPORTED_ONNX_OPSET = 22

# These are the immutable ONNX operator schema versions reached by models whose
# ai.onnx opset is in the supported range above. Any new schema version requires
# an explicit semantic review even when the operator name is already familiar.
SUPPORTED_OPERATOR_SCHEMAS = {
    "Abs": {13},
    "Acos": {7, 22},
    "Acosh": {9, 22},
    "Add": {13, 14},
    "And": {7},
    "Asin": {7, 22},
    "Asinh": {9, 22},
    "Atan": {7, 22},
    "Atanh": {9, 22},
    "BatchNormalization": {9, 14, 15},
    "Cast": {13, 19, 21},
    "CastLike": {15, 19, 21},
    "Ceil": {13},
    "Celu": {12},
    "Clip": {13},
    "Concat": {13},
    "Constant": {13, 19, 21},
    "Cos": {7, 22},
    "Cosh": {9, 22},
    "Div": {13, 14},
    "Dropout": {13, 22},
    "Elu": {6, 22},
    "Equal": {13, 19},
    "Erf": {13},
    "Exp": {13},
    "Flatten": {13, 21},
    "Floor": {13},
    "Gather": {13},
    "Gelu": {20},
    "Gemm": {13},
    "Greater": {13},
    "GreaterOrEqual": {12, 16},
    "HardSigmoid": {6, 22},
    "HardSwish": {14, 22},
    "Identity": {13, 14, 16, 19, 21},
    "IsInf": {10, 20},
    "IsNaN": {13, 20},
    "LayerNormalization": {17},
    "LeakyRelu": {6, 16},
    "Less": {13},
    "LessOrEqual": {12, 16},
    "Log": {13},
    "LogSoftmax": {13},
    "LpNormalization": {1, 22},
    "MatMul": {13},
    "Max": {13},
    "Mean": {13},
    "Min": {13},
    "Mish": {18, 22},
    "Mul": {13, 14},
    "Neg": {13},
    "Not": {1},
    "Or": {7},
    "Pow": {13, 15},
    "PRelu": {9, 16},
    "ReduceL1": {13, 18},
    "ReduceL2": {13, 18},
    "ReduceLogSum": {13, 18},
    "ReduceLogSumExp": {13, 18},
    "ReduceMax": {13, 18, 20},
    "ReduceMean": {13, 18},
    "ReduceMin": {13, 18, 20},
    "ReduceProd": {13, 18},
    "ReduceSum": {13},
    "ReduceSumSquare": {13, 18},
    "Reciprocal": {13},
    "Relu": {13, 14},
    "Reshape": {13, 14, 19, 21},
    "Round": {11, 22},
    "Selu": {6, 22},
    "Shape": {13, 15, 19, 21},
    "Sigmoid": {13},
    "Sign": {13},
    "Sin": {7, 22},
    "Sinh": {9, 22},
    "Size": {13, 19, 21},
    "Softmax": {13},
    "Softplus": {1, 22},
    "Softsign": {1, 22},
    "Sqrt": {13},
    "Sub": {13, 14},
    "Sum": {13},
    "Squeeze": {13, 21},
    "Tan": {7, 22},
    "Tanh": {13},
    "ThresholdedRelu": {10, 22},
    "Transpose": {13, 21},
    "Unsqueeze": {13, 21},
    "Where": {9, 16},
    "Xor": {7},
}
SUPPORTED_OPS = frozenset(SUPPORTED_OPERATOR_SCHEMAS)
UNARY_OPS = {
    "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos", "Cosh", "Elu", "Erf",
    "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish", "Neg", "Relu",
    "Reciprocal", "Round", "Selu", "Sigmoid", "Sign", "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Tan",
    "Tanh", "ThresholdedRelu",
}
BINARY_OPS = {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}
COMPARISON_OPS = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}
LOGICAL_BINARY_OPS = {"And", "Or", "Xor"}
REDUCTION_OPS = {
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin",
    "ReduceProd", "ReduceSum", "ReduceSumSquare",
}


def _onnx_modules():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise CompilerError("ONNX support is not installed; install src/generator/requirements.txt") from exc
    return onnx, numpy_helper


def _dtype(element_type: int) -> DType:
    onnx, _ = _onnx_modules()
    if element_type == onnx.TensorProto.BOOL:
        return DType.BOOL
    if element_type == onnx.TensorProto.FLOAT:
        return DType.FLOAT32
    if element_type == onnx.TensorProto.DOUBLE:
        return DType.FLOAT64
    if element_type == onnx.TensorProto.INT32:
        return DType.INT32
    if element_type == onnx.TensorProto.INT64:
        return DType.INT64
    raise CompilerError(
        f"unsupported tensor element type {element_type}; only bool, float, double, int32, and int64 are supported"
    )


def _attribute_value(attribute: Any) -> Any:
    onnx, numpy_helper = _onnx_modules()
    kind = attribute.type
    if kind == onnx.AttributeProto.INT:
        return int(attribute.i)
    if kind == onnx.AttributeProto.FLOAT:
        return float(attribute.f)
    if kind == onnx.AttributeProto.INTS:
        return [int(value) for value in attribute.ints]
    if kind == onnx.AttributeProto.FLOATS:
        return [float(value) for value in attribute.floats]
    if kind == onnx.AttributeProto.STRING:
        return attribute.s.decode("utf-8")
    if kind == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attribute.t)
    raise CompilerError(f"unsupported ONNX attribute type {kind} on attribute {attribute.name!r}")


def _canonical_domain(domain: str) -> str:
    return "ai.onnx" if domain in {"", "ai.onnx"} else domain


def _opset_versions(model: Any) -> dict[str, int]:
    versions: dict[str, int] = {}
    for imported in model.opset_import:
        domain = _canonical_domain(imported.domain)
        version = int(imported.version)
        if domain in versions and versions[domain] != version:
            raise CompilerError(
                f"model imports ONNX domain {domain!r} at conflicting versions {versions[domain]} and {version}"
            )
        versions[domain] = version
    if "ai.onnx" not in versions:
        raise CompilerError("model does not declare an ai.onnx operator-set version")
    version = versions["ai.onnx"]
    if not MIN_SUPPORTED_ONNX_OPSET <= version <= MAX_SUPPORTED_ONNX_OPSET:
        raise CompilerError(
            f"unsupported ai.onnx opset {version}; PONNI supports opsets "
            f"{MIN_SUPPORTED_ONNX_OPSET} through {MAX_SUPPORTED_ONNX_OPSET}"
        )
    return versions


def _validate_schema_arity(node: Any, schema: Any) -> None:
    name = node.name or node.op_type
    input_count = sum(bool(value) for value in node.input)
    output_count = sum(bool(value) for value in node.output)
    if not schema.min_input <= input_count <= schema.max_input:
        raise CompilerError(
            f"ONNX schema {node.op_type}:{schema.since_version} requires {schema.min_input}..{schema.max_input} "
            f"inputs, but node {name!r} has {input_count}"
        )
    if not schema.min_output <= output_count <= schema.max_output:
        raise CompilerError(
            f"ONNX schema {node.op_type}:{schema.since_version} requires {schema.min_output}..{schema.max_output} "
            f"outputs, but node {name!r} has {output_count}"
        )
    for index, formal in enumerate(schema.inputs):
        if str(formal.option).endswith(".Single") and (index >= len(node.input) or not node.input[index]):
            raise CompilerError(
                f"ONNX schema {node.op_type}:{schema.since_version} requires input {index} "
                f"({formal.name!r}) on node {name!r}"
            )
    for index, formal in enumerate(schema.outputs):
        if str(formal.option).endswith(".Single") and (index >= len(node.output) or not node.output[index]):
            raise CompilerError(
                f"ONNX schema {node.op_type}:{schema.since_version} requires output {index} "
                f"({formal.name!r}) on node {name!r}"
            )


def _schema_attributes(node: Any, opsets: dict[str, int]) -> tuple[dict[str, Any], int]:
    onnx, _ = _onnx_modules()
    domain = _canonical_domain(node.domain)
    if domain != "ai.onnx":
        raise CompilerError(f"node {node.name or node.op_type!r} uses unsupported ONNX domain {domain!r}")
    if node.op_type not in SUPPORTED_OPERATOR_SCHEMAS:
        raise CompilerError(
            f"node {node.name or node.op_type!r} uses unsupported operator {node.op_type!r}; "
            f"supported operators: {', '.join(sorted(SUPPORTED_OPS - {'Constant'}))}"
        )
    try:
        schema = onnx.defs.get_schema(node.op_type, opsets[domain], "")
    except Exception as exc:
        raise CompilerError(
            f"no ONNX schema for {domain}::{node.op_type} at opset {opsets[domain]}"
        ) from exc
    if schema.since_version not in SUPPORTED_OPERATOR_SCHEMAS[node.op_type]:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_OPERATOR_SCHEMAS[node.op_type]))
        raise CompilerError(
            f"unsupported ONNX schema {domain}::{node.op_type}:{schema.since_version}; "
            f"PONNI supports schema versions {supported}"
        )
    _validate_schema_arity(node, schema)
    attributes = {attribute.name: _attribute_value(attribute) for attribute in node.attribute}
    unknown = sorted(set(attributes) - set(schema.attributes))
    if unknown:
        raise CompilerError(
            f"node {node.name or node.op_type!r} has attributes outside ONNX schema "
            f"{node.op_type}:{schema.since_version}: {', '.join(unknown)}"
        )
    for name, specification in schema.attributes.items():
        if specification.required and name not in attributes:
            raise CompilerError(
                f"node {node.name or node.op_type!r} omits required attribute {name!r} from "
                f"ONNX schema {node.op_type}:{schema.since_version}"
            )
        default = specification.default_value
        if name not in attributes and default.type != onnx.AttributeProto.UNDEFINED:
            attributes[name] = _attribute_value(default)
    # These fields were introduced after opset 13 with defaults that preserve
    # the behavior of the older schemas.  Materialize them so validation and
    # lowering consume one canonical contract regardless of source opset.
    if node.op_type in REDUCTION_OPS:
        attributes.setdefault("noop_with_empty_axes", 0)
    elif node.op_type == "Reshape":
        attributes.setdefault("allowzero", 0)
    return attributes, int(schema.since_version)


def _shape(value_info: Any, batch_symbol: str, allow_derived_batch: bool = False) -> tuple[int | Symbol, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise CompilerError(f"tensor {value_info.name!r} has no inferred shape")
    result: list[int | Symbol] = []
    for axis, dimension in enumerate(tensor_type.shape.dim):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            if dimension.dim_param != batch_symbol and not allow_derived_batch:
                raise CompilerError(
                    f"tensor {value_info.name!r} axis {axis} has dynamic non-batch dimension "
                    f"{dimension.dim_param!r}; only the batch dimension may be dynamic"
                )
            result.append(Symbol(batch_symbol))
        else:
            raise CompilerError(f"tensor {value_info.name!r} axis {axis} has an unresolved dimension")
    return tuple(result)


def _value_info_map(model: Any) -> dict[str, Any]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    return {value.name: value for value in values}


def _fold_static_shape_nodes(graph: Graph) -> None:
    """Evaluate shape/index glue whose result is independent of the runtime batch size."""
    retained: list[Node] = []
    for node in graph.nodes:
        result: np.ndarray | None = None
        if node.op == "Shape":
            shape = graph.tensors[node.inputs[0]].shape
            start = int(node.attributes.get("start", 0))
            end = int(node.attributes.get("end", len(shape)))
            start = start + len(shape) if start < 0 else start
            end = end + len(shape) if end < 0 else end
            selected = shape[max(0, start):min(len(shape), end)]
            if all(isinstance(dim, int) for dim in selected):
                result = np.asarray(selected, dtype=np.int64)
        elif node.op == "Size":
            shape = graph.tensors[node.inputs[0]].shape
            if all(isinstance(dim, int) for dim in shape):
                result = np.asarray(math.prod(shape), dtype=np.int64)
        elif node.op == "Gather":
            data = graph.tensors[node.inputs[0]]
            if data.is_constant and data.constant_name is not None:
                values = graph.constants[data.constant_name].values
                result = np.take(values, node.attributes["indices"], axis=int(node.attributes.get("axis", 0)))
        elif node.op in {"Squeeze", "Unsqueeze"}:
            data = graph.tensors[node.inputs[0]]
            if data.is_constant and data.constant_name is not None:
                result = graph.constants[data.constant_name].values
                axes = [int(axis) for axis in node.attributes.get("axes", [])]
                for axis in sorted(axes, reverse=node.op == "Squeeze"):
                    result = np.squeeze(result, axis=axis) if node.op == "Squeeze" else np.expand_dims(result, axis)
        elif node.op == "Concat":
            tensors = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            if tensors and all(tensor.is_constant and tensor.constant_name is not None for tensor in tensors):
                values = [graph.constants[tensor.constant_name].values for tensor in tensors]
                result = np.concatenate(values, axis=int(node.attributes.get("axis", 0)))
        if result is None:
            retained.append(node)
            continue
        output = graph.tensors[node.outputs[0]]
        values = np.asarray(result)
        constant_name = f"__static_{node.outputs[0]}_{output.name}"
        graph.constants[constant_name] = ConstantTensor(
            constant_name, tuple(int(dim) for dim in values.shape), output.dtype, values.copy(), "static-shape", False
        )
        output.is_constant = True
        output.constant_name = constant_name
        output.shape = tuple(int(dim) for dim in values.shape)
    graph.nodes = retained
    graph.renumber_nodes()


def import_onnx(path: str | Path | Any) -> Graph:
    onnx, numpy_helper = _onnx_modules()
    model_path = Path(path) if isinstance(path, (str, Path)) else None
    try:
        model = onnx.load(model_path) if model_path is not None else path
        if not MIN_SUPPORTED_IR_VERSION <= model.ir_version <= MAX_SUPPORTED_IR_VERSION:
            raise CompilerError(
                f"unsupported ONNX IR version {model.ir_version}; PONNI supports IR versions "
                f"{MIN_SUPPORTED_IR_VERSION} through {MAX_SUPPORTED_IR_VERSION}"
            )
        opsets = _opset_versions(model)
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(model)
    except CompilerError:
        raise
    except Exception as exc:
        source = model_path if model_path is not None else "in-memory model"
        raise CompilerError(f"ONNX validation or shape inference failed for {source}: {exc}") from exc

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    batch_symbol = metadata.get("ponni.batch_symbol", "batch")
    orientation = metadata.get("ponni.orientation", "features_batch")
    if orientation != "features_batch":
        raise CompilerError(
            f"unsupported logical orientation {orientation!r}; expected feature-major (num_features, batch_size)"
        )

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(graph_inputs) != 1 or len(model.graph.output) != 1:
        raise CompilerError(
            f"the prototype requires exactly one model input and one output; found {len(graph_inputs)} and "
            f"{len(model.graph.output)}"
        )

    schema_semantics = [_schema_attributes(node, opsets) for node in model.graph.node]

    values = _value_info_map(model)
    constants: dict[str, ConstantTensor] = {}
    for initializer in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(initializer))
        constants[initializer.name] = ConstantTensor(
            initializer.name, tuple(int(dim) for dim in array.shape), _dtype(initializer.data_type), array.copy()
        )

    # Convert ONNX Constant nodes into initializers before assigning tensor IDs.
    constant_outputs: set[str] = set()
    for node, (attributes, _) in zip(model.graph.node, schema_semantics):
        if node.op_type != "Constant":
            continue
        if "value" not in attributes or len(node.output) != 1:
            raise CompilerError(f"Constant node {node.name!r} must contain one tensor-valued 'value' attribute")
        array = np.asarray(attributes["value"])
        if array.dtype not in (np.bool_, np.float32, np.float64, np.int32, np.int64):
            raise CompilerError(f"Constant node {node.name!r} has unsupported dtype {array.dtype}")
        dtype = {
            np.dtype(np.bool_): DType.BOOL,
            np.dtype(np.float32): DType.FLOAT32,
            np.dtype(np.float64): DType.FLOAT64,
            np.dtype(np.int32): DType.INT32,
            np.dtype(np.int64): DType.INT64,
        }[array.dtype]
        constants[node.output[0]] = ConstantTensor(
            node.output[0], tuple(int(dim) for dim in array.shape), dtype, array.copy()
        )
        constant_outputs.add(node.output[0])

    all_names: list[str] = []
    all_names.extend(value.name for value in graph_inputs)
    all_names.extend(initializer.name for initializer in model.graph.initializer)
    for node in model.graph.node:
        all_names.extend(name for name in node.input if name)
        all_names.extend(name for name in node.output if name)
    all_names.extend(value.name for value in model.graph.output)
    ordered_names = list(dict.fromkeys(all_names))
    name_to_id = {name: tensor_id for tensor_id, name in enumerate(ordered_names)}

    input_names = {value.name for value in graph_inputs}
    output_names = {value.name for value in model.graph.output}
    tensors: dict[int, TensorValue] = {}
    for name in ordered_names:
        tensor_id = name_to_id[name]
        if name in constants:
            constant = constants[name]
            tensor_shape: tuple[int | Symbol, ...] = constant.shape
            tensor_dtype = constant.dtype
        else:
            if name not in values:
                raise CompilerError(f"shape inference produced no type/shape information for tensor {name!r}")
            tensor_shape = _shape(values[name], batch_symbol, name not in input_names | output_names)
            tensor_dtype = _dtype(values[name].type.tensor_type.elem_type)
        tensors[tensor_id] = TensorValue(
            tensor_id,
            name,
            tensor_shape,
            tensor_dtype,
            is_input=name in input_names,
            is_output=name in output_names,
            is_constant=name in constants,
            constant_name=name if name in constants else None,
        )

    nodes: list[Node] = []
    schema_counts: Counter[str] = Counter()
    for onnx_node, (attributes, schema_version) in zip(model.graph.node, schema_semantics):
        schema_counts[f"{onnx_node.op_type}:{schema_version}"] += 1
        if onnx_node.op_type == "Constant":
            continue
        input_names = list(onnx_node.input)
        if onnx_node.op_type == "Clip":
            for index, attribute_name in ((1, "min"), (2, "max")):
                if index >= len(input_names) or not input_names[index]:
                    continue
                bound_name = input_names[index]
                if bound_name not in constants or constants[bound_name].values.size != 1:
                    raise CompilerError(
                        f"Clip {onnx_node.name or onnx_node.op_type!r} requires constant scalar {attribute_name}"
                    )
                attributes[attribute_name] = float(constants[bound_name].values.item())
            input_names = input_names[:1]
        elif onnx_node.op_type in REDUCTION_OPS and len(input_names) > 1 and input_names[1]:
            axes_name = input_names[1]
            if axes_name not in constants:
                raise CompilerError(
                    f"{onnx_node.op_type} {onnx_node.name or onnx_node.op_type!r} requires constant axes"
                )
            attributes["axes"] = [int(value) for value in constants[axes_name].values.reshape(-1)]
            input_names = input_names[:1]
        elif onnx_node.op_type in {"Squeeze", "Unsqueeze"} and len(input_names) > 1 and input_names[1]:
            axes_name = input_names[1]
            if axes_name not in constants:
                raise CompilerError(
                    f"{onnx_node.op_type} {onnx_node.name or onnx_node.op_type!r} requires constant axes"
                )
            attributes["axes"] = [int(value) for value in constants[axes_name].values.reshape(-1)]
            input_names = input_names[:1]
        elif onnx_node.op_type == "Gather":
            if len(input_names) != 2 or input_names[1] not in constants:
                raise CompilerError(
                    f"Gather {onnx_node.name or onnx_node.op_type!r} requires compile-time constant indices"
                )
            attributes["indices"] = np.asarray(constants[input_names[1]].values).copy()
            input_names = input_names[:1]
        elif onnx_node.op_type == "Dropout":
            if len([name for name in onnx_node.output if name]) != 1:
                raise CompilerError(
                    f"Dropout {onnx_node.name or onnx_node.op_type!r} supports only the primary inference output"
                )
            if len(input_names) > 2 and input_names[2]:
                training_name = input_names[2]
                if training_name not in constants or bool(constants[training_name].values.item()):
                    raise CompilerError(
                        f"Dropout {onnx_node.name or onnx_node.op_type!r} supports only inference mode"
                    )
            input_names = input_names[:1]
            attributes["inference_only"] = True
        canonical_op = "Identity" if onnx_node.op_type == "Dropout" else onnx_node.op_type
        nodes.append(
            Node(
                len(nodes),
                canonical_op,
                [name_to_id[name] for name in input_names if name],
                [name_to_id[name] for name in onnx_node.output if name],
                attributes,
                onnx_node.name,
            )
        )

    graph = Graph(
        [name_to_id[value.name] for value in graph_inputs],
        [name_to_id[value.name] for value in model.graph.output],
        tensors,
        nodes,
        constants,
        {
            "source": str(model_path),
            "ir_version": int(model.ir_version),
            "opset": opsets["ai.onnx"],
            "opsets": dict(sorted(opsets.items())),
            "operator_schema_counts": dict(sorted(schema_counts.items())),
            "orientation": orientation,
            "batch_symbol": batch_symbol,
            "original_node_count": len(model.graph.node),
            "operator_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        },
    )
    graph.rebuild_links()
    for node in graph.nodes:
        if node.op != "CastLike":
            continue
        source = graph.tensors[node.inputs[0]]
        target = graph.tensors[node.inputs[1]]
        output = graph.tensors[node.outputs[0]]
        if source.dtype == target.dtype == output.dtype:
            node.op = "Identity"
        elif source.dtype == DType.BOOL and target.dtype == output.dtype and output.dtype in {
                DType.FLOAT32, DType.FLOAT64}:
            node.op = "Cast"
        else:
            raise CompilerError(
                f"CastLike {node.source_name or node.op!r} supports only no-op casts and Boolean-to-floating casts"
            )
        node.inputs = node.inputs[:1]
    _fold_static_shape_nodes(graph)
    for node in graph.nodes:
        if node.op == "Transpose" and "perm" not in node.attributes:
            rank = len(graph.tensors[node.inputs[0]].shape)
            node.attributes["perm"] = list(reversed(range(rank)))
    # ONNX does not carry a general requires-gradient flag. Mark only constants in
    # well-defined learned roles. Literal constants, shape/axis inputs, clipping
    # bounds, and BatchNormalization running statistics remain static.
    for node in graph.nodes:
        learned_inputs: list[int] = []
        if node.op == "Gemm":
            learned_inputs = node.inputs[1:3]
        elif node.op == "MatMul" and len(node.inputs) == 2:
            learned_inputs = [node.inputs[1]]
        elif node.op == "LayerNormalization":
            learned_inputs = node.inputs[1:3]
        elif node.op == "BatchNormalization":
            learned_inputs = node.inputs[1:3]
        elif node.op == "PRelu":
            learned_inputs = node.inputs[1:2]
        for tensor_id in learned_inputs:
            tensor = graph.tensors[tensor_id]
            if tensor.is_constant and tensor.constant_name is not None:
                graph.constants[tensor.constant_name].learned = True

    # A constant Add immediately following a matrix product is the conventional
    # dense bias spelling used by PyTorch, Keras, and TensorFlow exporters.
    graph.rebuild_links()
    for node in graph.nodes:
        if node.op != "Add":
            continue
        has_matrix_product = any(
            graph.tensors[tensor_id].producer is not None and
            graph.node_by_id(graph.tensors[tensor_id].producer).op in {"Gemm", "MatMul"}
            for tensor_id in node.inputs
        )
        if not has_matrix_product:
            continue
        for tensor_id in node.inputs:
            tensor = graph.tensors[tensor_id]
            if tensor.is_constant and tensor.constant_name is not None:
                graph.constants[tensor.constant_name].learned = True
    validate_graph(graph)
    return graph


def validate_graph(graph: Graph) -> None:
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise CompilerError("canonical graph requires exactly one input and one output")
    for tensor_id in graph.inputs + graph.outputs:
        tensor = graph.tensors[tensor_id]
        if tensor.dtype not in {DType.FLOAT32, DType.FLOAT64}:
            raise CompilerError(
                f"model boundary tensor {tensor.name!r} has unsupported dtype {tensor.dtype.value}; "
                "PONNI supports float32 and float64 model boundaries"
            )
    for node in graph.nodes:
        output = graph.tensors[node.outputs[0]] if node.outputs else None
        if node.op in UNARY_OPS:
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires one input and one output")
            if graph.tensors[node.inputs[0]].sample_size != output.sample_size:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} must preserve per-sample shape")
        if node.op == "Gelu" and str(node.attributes.get("approximate", "none")) not in {"none", "tanh"}:
            raise CompilerError(
                f"Gelu {node.source_name or node.op!r} has unsupported approximate mode "
                f"{node.attributes.get('approximate')!r}"
            )
        if node.op in {"Softmax", "LogSoftmax", "LayerNormalization"}:
            if node.op in {"Softmax", "LogSoftmax"} and (len(node.inputs) != 1 or len(node.outputs) != 1):
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires one input and one output")
            data = graph.tensors[node.inputs[0]]
            if len(data.shape) != 2 or output is None or len(output.shape) != 2:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} supports only rank-two tensors")
            batch_axes = [axis for axis, dim in enumerate(data.shape) if isinstance(dim, Symbol)]
            if len(batch_axes) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires one dynamic batch axis")
            axis = int(node.attributes.get("axis", -1))
            if axis < 0:
                axis += len(data.shape)
            if axis == batch_axes[0] or axis < 0 or axis >= len(data.shape):
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} must operate over the static feature axis"
                )
        if node.op == "LayerNormalization":
            if int(node.attributes["stash_type"]) != 1:
                raise CompilerError(
                    f"LayerNormalization {node.source_name or node.op!r} supports only stash_type=1"
                )
            if len(node.inputs) not in (2, 3) or len(node.outputs) != 1:
                raise CompilerError(
                    f"LayerNormalization {node.source_name or node.op!r} requires data, scale, optional bias, "
                    "and one output"
                )
            feature_size = graph.tensors[node.inputs[0]].sample_size
            for tensor_id in node.inputs[1:]:
                tensor = graph.tensors[tensor_id]
                if not tensor.is_constant or tensor.sample_size not in (1, feature_size):
                    raise CompilerError(
                        f"LayerNormalization {node.source_name or node.op!r} requires constant scalar or "
                        "feature-sized scale and bias"
                    )
        if node.op == "BatchNormalization":
            if len(node.inputs) != 5 or len(node.outputs) != 1 or int(node.attributes.get("training_mode", 0)) != 0:
                raise CompilerError(
                    f"BatchNormalization {node.source_name or node.op!r} supports only five-input inference mode"
                )
            feature_size = graph.tensors[node.inputs[0]].sample_size
            for tensor_id in node.inputs[1:]:
                tensor = graph.tensors[tensor_id]
                if not tensor.is_constant or tensor.sample_size not in (1, feature_size):
                    raise CompilerError(
                        f"BatchNormalization {node.source_name or node.op!r} parameters must be constant scalar "
                        "or feature-sized tensors"
                    )
        if node.op == "Clip":
            if len(node.inputs) != 1:
                raise CompilerError(f"Clip {node.source_name or node.op!r} must have one canonical data input")
        if node.op in REDUCTION_OPS:
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} has unsupported input/output count")
            if int(node.attributes["keepdims"]) not in (0, 1):
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} requires keepdims=0 or keepdims=1"
                )
            data = graph.tensors[node.inputs[0]]
            if len(data.shape) != 2 or output is None or output.sample_size != 1:
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} supports only full feature-axis reduction"
                )
            axes = node.attributes.get("axes")
            if axes is None or not axes:
                if int(node.attributes.get("noop_with_empty_axes", 0)) == 1:
                    node.op = "Identity"
                    continue
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires an explicit feature axis")
            normalized = [int(axis) + len(data.shape) if int(axis) < 0 else int(axis) for axis in axes]
            batch_axis = next(axis for axis, dim in enumerate(data.shape) if isinstance(dim, Symbol))
            if normalized != [1 - batch_axis]:
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} may reduce only the static feature axis"
                )
        if node.op == "LpNormalization":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(
                    f"LpNormalization {node.source_name or node.op!r} requires one input and one output"
                )
            if int(node.attributes.get("p", 2)) not in (1, 2):
                raise CompilerError(f"LpNormalization {node.source_name or node.op!r} supports only p=1 or p=2")
            data = graph.tensors[node.inputs[0]]
            batch_axis = next(axis for axis, dim in enumerate(data.shape) if isinstance(dim, Symbol))
            axis = int(node.attributes.get("axis", -1))
            axis = axis + len(data.shape) if axis < 0 else axis
            if len(data.shape) != 2 or axis != 1 - batch_axis or output.sample_size != data.sample_size:
                raise CompilerError(
                    f"LpNormalization {node.source_name or node.op!r} supports only the static feature axis"
                )
        if node.op == "Reshape":
            if int(node.attributes["allowzero"]) != 0:
                raise CompilerError(f"Reshape {node.source_name or node.op!r} supports only allowzero=0")
            if len(node.inputs) != 2 or not graph.tensors[node.inputs[1]].is_constant:
                raise CompilerError(
                    f"Reshape {node.source_name or node.op!r} requires a compile-time constant shape tensor"
                )
        if node.op == "Concat":
            if len(node.inputs) < 2 or len(node.outputs) != 1:
                raise CompilerError(f"Concat {node.source_name or node.op!r} requires at least two inputs")
            tensors = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            output = graph.tensors[node.outputs[0]]
            if any(len(tensor.shape) != 2 for tensor in tensors) or len(output.shape) != 2:
                raise CompilerError(
                    f"Concat {node.source_name or node.op!r} supports only rank-two batch/feature tensors"
                )
            if any(tensor.dtype != output.dtype for tensor in tensors):
                raise CompilerError(f"Concat {node.source_name or node.op!r} requires matching floating-point types")
            dynamic_axes = [
                [axis for axis, dim in enumerate(tensor.shape) if isinstance(dim, Symbol)] for tensor in tensors
            ]
            output_dynamic = [axis for axis, dim in enumerate(output.shape) if isinstance(dim, Symbol)]
            if (any(len(axes) != 1 for axes in dynamic_axes) or len(output_dynamic) != 1 or
                    any(axes[0] != output_dynamic[0] for axes in dynamic_axes)):
                raise CompilerError(
                    f"Concat {node.source_name or node.op!r} must preserve one common batch axis"
                )
            axis = int(node.attributes.get("axis", 0))
            if axis < 0:
                axis += len(output.shape)
            if axis == output_dynamic[0]:
                raise CompilerError(
                    f"Concat {node.source_name or node.op!r} joins the runtime batch axis; only static feature-axis "
                    "concatenation is supported"
                )
            input_size = sum(tensor.sample_size for tensor in tensors)
            if input_size != output.sample_size:
                raise CompilerError(
                    f"Concat {node.source_name or node.op!r} has {input_size} input sample elements but "
                    f"{output.sample_size} output elements"
                )
        if node.op in {"Shape", "Size"}:
            raise CompilerError(
                f"{node.op} {node.source_name or node.op!r} depends on the runtime batch size; only statically "
                "resolvable shape expressions are supported"
            )
        if node.op in {"Squeeze", "Unsqueeze"}:
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires one canonical data input")
            data = graph.tensors[node.inputs[0]]
            if data.sample_size != output.sample_size:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} must preserve sample element count")
        if node.op == "Gather":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"Gather {node.source_name or node.op!r} requires one canonical data input")
            data = graph.tensors[node.inputs[0]]
            axis = int(node.attributes.get("axis", 0))
            axis = axis + len(data.shape) if axis < 0 else axis
            batch_axis = next(axis for axis, dim in enumerate(data.shape) if isinstance(dim, Symbol))
            indices = np.asarray(node.attributes["indices"])
            if len(data.shape) != 2 or axis != 1 - batch_axis or indices.ndim > 1:
                raise CompilerError(
                    f"Gather {node.source_name or node.op!r} supports scalar or vector constant indices on the "
                    "static feature axis"
                )
            normalized = np.where(indices < 0, indices + data.sample_size, indices)
            if np.any(normalized < 0) or np.any(normalized >= data.sample_size):
                raise CompilerError(f"Gather {node.source_name or node.op!r} has an out-of-range feature index")
            node.attributes["indices"] = [int(value) for value in normalized.reshape(-1)]
            if len(node.attributes["indices"]) != output.sample_size:
                raise CompilerError(f"Gather {node.source_name or node.op!r} has inconsistent output size")
        if node.op == "PRelu":
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise CompilerError(f"PRelu {node.source_name or node.op!r} requires data and slope inputs")
            data, slope = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            if data.dtype != output.dtype or slope.dtype != output.dtype:
                raise CompilerError(f"PRelu {node.source_name or node.op!r} requires matching floating-point types")
            if data.sample_size != output.sample_size or slope.sample_size not in (1, output.sample_size):
                raise CompilerError(f"PRelu {node.source_name or node.op!r} has unsupported broadcasting")
        if node.op in {"Mean", "Sum"}:
            if len(node.inputs) < 1 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires at least one input")
            tensors = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            if any(tensor.dtype != output.dtype for tensor in tensors):
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires matching input types")
            if any(tensor.sample_size not in (1, output.sample_size) for tensor in tensors):
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} has unsupported broadcasting")
        if node.op in {"IsInf", "IsNaN"}:
            if len(node.inputs) != 1 or len(node.outputs) != 1 or output.dtype != DType.BOOL:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires floating input and Boolean output")
            data = graph.tensors[node.inputs[0]]
            if data.dtype not in {DType.FLOAT32, DType.FLOAT64} or data.sample_size != output.sample_size:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} must preserve per-sample shape")
        if node.op in COMPARISON_OPS:
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires two inputs and one output")
            inputs = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            if output is None or output.dtype != DType.BOOL:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} must produce a Boolean tensor")
            allowed = {DType.FLOAT32, DType.FLOAT64, DType.BOOL} if node.op == "Equal" else {
                DType.FLOAT32, DType.FLOAT64,
            }
            if inputs[0].dtype != inputs[1].dtype or inputs[0].dtype not in allowed:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} has unsupported input types")
            if any(tensor.sample_size not in (1, output.sample_size) for tensor in inputs):
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} supports only scalar and exact-shape broadcasting"
                )
        if node.op in LOGICAL_BINARY_OPS:
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires two inputs and one output")
            tensors = [graph.tensors[tensor_id] for tensor_id in node.inputs + node.outputs]
            if any(tensor.dtype != DType.BOOL for tensor in tensors):
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires Boolean tensors")
            if any(tensor.sample_size not in (1, output.sample_size) for tensor in tensors[:2]):
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} supports only scalar and exact-shape broadcasting"
                )
        if node.op == "Not":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"Not {node.source_name or node.op!r} requires one input and one output")
            if graph.tensors[node.inputs[0]].dtype != DType.BOOL or output is None or output.dtype != DType.BOOL:
                raise CompilerError(f"Not {node.source_name or node.op!r} requires Boolean tensors")
            if graph.tensors[node.inputs[0]].sample_size != output.sample_size:
                raise CompilerError(f"Not {node.source_name or node.op!r} must preserve per-sample shape")
        if node.op == "Cast":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompilerError(f"Cast {node.source_name or node.op!r} requires one input and one output")
            input_tensor = graph.tensors[node.inputs[0]]
            if (input_tensor.dtype != DType.BOOL or output is None or
                    output.dtype not in {DType.FLOAT32, DType.FLOAT64}):
                raise CompilerError(
                    f"Cast {node.source_name or node.op!r} currently supports only Boolean-to-floating conversion"
                )
            if input_tensor.sample_size != output.sample_size:
                raise CompilerError(f"Cast {node.source_name or node.op!r} must preserve per-sample shape")
        if node.op == "Where":
            if len(node.inputs) != 3 or len(node.outputs) != 1:
                raise CompilerError(f"Where {node.source_name or node.op!r} requires three inputs and one output")
            condition, when_true, when_false = [graph.tensors[tensor_id] for tensor_id in node.inputs]
            if condition.dtype != DType.BOOL:
                raise CompilerError(f"Where {node.source_name or node.op!r} requires a Boolean condition")
            if output is None or when_true.dtype != when_false.dtype or output.dtype != when_true.dtype:
                raise CompilerError(f"Where {node.source_name or node.op!r} requires matching branch/output types")
            if output.dtype not in {DType.FLOAT32, DType.FLOAT64}:
                raise CompilerError(f"Where {node.source_name or node.op!r} currently supports floating-point branches")
            if any(tensor.sample_size not in (1, output.sample_size)
                   for tensor in (condition, when_true, when_false)):
                raise CompilerError(
                    f"Where {node.source_name or node.op!r} supports only scalar and exact-shape broadcasting"
                )
        if node.op not in BINARY_OPS:
            continue
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise CompilerError(
                f"{node.op} {node.source_name or node.op!r} requires exactly two inputs and one output"
            )
        input_sizes = [graph.tensors[tensor_id].sample_size for tensor_id in node.inputs]
        output_size = graph.tensors[node.outputs[0]].sample_size
        if any(size not in (1, output_size) for size in input_sizes):
            raise CompilerError(
                f"unsupported {node.op} broadcasting at {node.source_name or node.op!r}: input per-sample sizes "
                f"{input_sizes}, inferred output size {output_size}; only scalar and exact-shape broadcasts are supported"
            )
    input_tensor = graph.tensors[graph.inputs[0]]
    output_tensor = graph.tensors[graph.outputs[0]]
    for role, tensor in (("input", input_tensor), ("output", output_tensor)):
        symbols = [axis for axis, dim in enumerate(tensor.shape) if isinstance(dim, Symbol)]
        if len(tensor.shape) != 2 or len(symbols) != 1:
            raise CompilerError(
                f"model {role} {tensor.name!r} must be rank two with one dynamic batch dimension; got {tensor.shape}"
            )
        if symbols[0] != 1:
            raise CompilerError(
                f"model {role} {tensor.name!r} must use logical (features, batch) orientation; "
                f"batch is axis {symbols[0]}"
            )
    for tensor in graph.tensors.values():
        if tensor.is_constant:
            continue
        dynamic_axes = [dim for dim in tensor.shape if isinstance(dim, Symbol)]
        if len(dynamic_axes) > 1:
            raise CompilerError(f"tensor {tensor.name!r} contains more than one dynamic dimension: {tensor.shape}")
        for dim in tensor.sample_shape:
            if dim <= 0:
                raise CompilerError(f"tensor {tensor.name!r} has non-positive dimension {dim}")
        if tensor.dtype not in (DType.BOOL, DType.FLOAT32, DType.FLOAT64):
            raise CompilerError(f"tensor {tensor.name!r} has unsupported dtype {tensor.dtype}")
