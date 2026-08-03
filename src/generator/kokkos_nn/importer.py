from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CompilerError
from .ir import ConstantTensor, DType, Graph, Node, Symbol, TensorValue


SUPPORTED_OPS = {
    "Abs", "Add", "BatchNormalization", "Clip", "Concat", "Constant", "Div", "Elu", "Exp", "Flatten",
    "Gelu", "Gemm", "HardSigmoid", "HardSwish", "Identity", "LayerNormalization", "LeakyRelu", "Log",
    "LogSoftmax", "MatMul", "Max", "Min", "Mish", "Mul", "Neg", "Pow", "ReduceMean", "ReduceSum",
    "Reciprocal", "Relu", "Reshape", "Sigmoid", "Softmax", "Softplus", "Sqrt", "Sub", "Tanh", "Transpose",
}
SUPPORTED_DOMAINS = {"", "ai.onnx"}
UNARY_OPS = {
    "Abs", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish", "Neg", "Relu",
    "Reciprocal", "Sigmoid", "Softplus", "Sqrt", "Tanh",
}
BINARY_OPS = {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}


def _onnx_modules():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise CompilerError("ONNX support is not installed; install src/generator/requirements.txt") from exc
    return onnx, numpy_helper


def _dtype(element_type: int) -> DType:
    onnx, _ = _onnx_modules()
    if element_type == onnx.TensorProto.FLOAT:
        return DType.FLOAT32
    if element_type == onnx.TensorProto.DOUBLE:
        return DType.FLOAT64
    if element_type == onnx.TensorProto.INT32:
        return DType.INT32
    if element_type == onnx.TensorProto.INT64:
        return DType.INT64
    raise CompilerError(f"unsupported tensor element type {element_type}; only float and double are supported")


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


def import_onnx(path: str | Path) -> Graph:
    onnx, numpy_helper = _onnx_modules()
    model_path = Path(path)
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise CompilerError(f"ONNX validation or shape inference failed for {model_path}: {exc}") from exc

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

    for node in model.graph.node:
        domain = node.domain or ""
        if domain not in SUPPORTED_DOMAINS:
            raise CompilerError(f"node {node.name or node.op_type!r} uses unsupported ONNX domain {domain!r}")
        if node.op_type not in SUPPORTED_OPS:
            raise CompilerError(
                f"node {node.name or node.op_type!r} uses unsupported operator {node.op_type!r}; "
                f"supported operators: {', '.join(sorted(SUPPORTED_OPS - {'Constant'}))}"
            )

    values = _value_info_map(model)
    constants: dict[str, ConstantTensor] = {}
    for initializer in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(initializer))
        constants[initializer.name] = ConstantTensor(
            initializer.name, tuple(int(dim) for dim in array.shape), _dtype(initializer.data_type), array.copy()
        )

    # Convert ONNX Constant nodes into initializers before assigning tensor IDs.
    constant_outputs: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        attributes = {attribute.name: _attribute_value(attribute) for attribute in node.attribute}
        if "value" not in attributes or len(node.output) != 1:
            raise CompilerError(f"Constant node {node.name!r} must contain one tensor-valued 'value' attribute")
        array = np.asarray(attributes["value"])
        if array.dtype not in (np.float32, np.float64, np.int32, np.int64):
            raise CompilerError(f"Constant node {node.name!r} has unsupported dtype {array.dtype}")
        dtype = {
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
    for onnx_node in model.graph.node:
        if onnx_node.op_type == "Constant":
            continue
        attributes = {attribute.name: _attribute_value(attribute) for attribute in onnx_node.attribute}
        nodes.append(
            Node(
                len(nodes),
                onnx_node.op_type,
                [name_to_id[name] for name in onnx_node.input if name],
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
            "opset": max((opset.version for opset in model.opset_import), default=0),
            "orientation": orientation,
            "batch_symbol": batch_symbol,
            "original_node_count": len(model.graph.node),
            "operator_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        },
    )
    graph.rebuild_links()
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
            if len(node.inputs) not in (1, 2, 3):
                raise CompilerError(f"Clip {node.source_name or node.op!r} has unsupported input count")
            for tensor_id in node.inputs[1:]:
                tensor = graph.tensors[tensor_id]
                if not tensor.is_constant or tensor.sample_size != 1:
                    raise CompilerError(f"Clip {node.source_name or node.op!r} requires constant scalar bounds")
        if node.op in {"ReduceMean", "ReduceSum"}:
            if len(node.inputs) not in (1, 2) or len(node.outputs) != 1:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} has unsupported input/output count")
            data = graph.tensors[node.inputs[0]]
            if len(data.shape) != 2 or output is None or output.sample_size != 1:
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} supports only full feature-axis reduction"
                )
            axes = node.attributes.get("axes")
            if len(node.inputs) == 2:
                axes_tensor = graph.tensors[node.inputs[1]]
                if not axes_tensor.is_constant or axes_tensor.constant_name is None:
                    raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires constant axes")
                axes = graph.constants[axes_tensor.constant_name].values.reshape(-1).tolist()
            if axes is None:
                raise CompilerError(f"{node.op} {node.source_name or node.op!r} requires an explicit feature axis")
            normalized = [int(axis) + len(data.shape) if int(axis) < 0 else int(axis) for axis in axes]
            batch_axis = next(axis for axis, dim in enumerate(data.shape) if isinstance(dim, Symbol))
            if normalized != [1 - batch_axis]:
                raise CompilerError(
                    f"{node.op} {node.source_name or node.op!r} may reduce only the static feature axis"
                )
        if node.op == "Reshape":
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
        if tensor.dtype not in (DType.FLOAT32, DType.FLOAT64):
            raise CompilerError(f"tensor {tensor.name!r} has unsupported dtype {tensor.dtype}")
