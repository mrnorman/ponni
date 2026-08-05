"""Deterministic canonicalization and fusion passes over PONNI IR.

The ordered pipeline exposes dense structure, recognizes reviewed exporter
decompositions, and finally forms compound regions for efficient emission.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable

import numpy as np

from .errors import CompilerError
from .ir import ConstantTensor, DType, Graph, Node


# These families define which operations are safe to embed in fused pointwise
# programs without changing evaluation order or broadcasting semantics.
ACTIVATIONS = {
    "Elu", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Mish", "Relu", "Sigmoid", "Silu",
    "Softplus", "Tanh",
}
ELEMENTWISE = {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}
COMPARISONS = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}
LOGICAL = {"And", "Or", "Xor"}
UNARY = ACTIVATIONS | {
    "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos", "Cosh", "Erf", "Exp",
    "Floor", "Log", "Neg", "Reciprocal", "Round", "Selu", "Sign", "Sin", "Sinh", "Softsign", "Sqrt", "Tan",
    "ThresholdedRelu",
}
POINTWISE = ELEMENTWISE | UNARY | COMPARISONS | LOGICAL | {
    "BatchNormalization", "Cast", "Clip", "CompareSelect", "IsInf", "IsNaN", "Mean", "Not", "PRelu",
    "Sum", "Where",
}


def _constant(graph: Graph, tensor_id: int) -> np.ndarray | None:
    tensor = graph.tensors[tensor_id]
    if not tensor.is_constant or tensor.constant_name is None:
        return None
    return graph.constants[tensor.constant_name].values


def _replace_tensor(graph: Graph, old_id: int, new_id: int) -> None:
    for node in graph.nodes:
        node.inputs = [new_id if tensor_id == old_id else tensor_id for tensor_id in node.inputs]
    graph.outputs = [new_id if tensor_id == old_id else tensor_id for tensor_id in graph.outputs]
    graph.tensors[new_id].is_output = graph.tensors[new_id].is_output or graph.tensors[old_id].is_output
    graph.tensors.pop(old_id, None)


def topological_schedule(graph: Graph) -> bool:
    """Restore deterministic topological order after graph rewrites."""
    graph.rebuild_links()
    node_map = {node.id: node for node in graph.nodes}
    indegree = {node.id: 0 for node in graph.nodes}
    successors: dict[int, set[int]] = defaultdict(set)
    for node in graph.nodes:
        predecessors: set[int] = set()
        for tensor_id in node.inputs:
            producer = graph.tensors[tensor_id].producer
            if producer is not None and producer != node.id:
                predecessors.add(producer)
        indegree[node.id] = len(predecessors)
        for producer in predecessors:
            successors[producer].add(node.id)
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    ordered: list[Node] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_map[node_id])
        for successor in sorted(successors[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered) != len(graph.nodes):
        raise CompilerError("graph contains a cycle; loops and cyclic dependencies are unsupported")
    changed = [node.id for node in ordered] != [node.id for node in graph.nodes]
    graph.nodes = ordered
    graph.renumber_nodes()
    return changed


def constant_fold(graph: Graph) -> bool:
    """Evaluate operations whose inputs are all compile-time constants."""
    changed = False
    retained: list[Node] = []
    for node in graph.nodes:
        inputs = [_constant(graph, tensor_id) for tensor_id in node.inputs]
        if not inputs or any(value is None for value in inputs):
            retained.append(node)
            continue
        values = [np.asarray(value) for value in inputs]
        try:
            if node.op == "Add":
                result = values[0] + values[1]
            elif node.op == "Sub":
                result = values[0] - values[1]
            elif node.op == "Mul":
                result = values[0] * values[1]
            elif node.op == "Div":
                result = values[0] / values[1]
            elif node.op == "Equal":
                result = np.equal(values[0], values[1])
            elif node.op == "Greater":
                result = np.greater(values[0], values[1])
            elif node.op == "GreaterOrEqual":
                result = np.greater_equal(values[0], values[1])
            elif node.op == "Less":
                result = np.less(values[0], values[1])
            elif node.op == "LessOrEqual":
                result = np.less_equal(values[0], values[1])
            elif node.op == "And":
                result = np.logical_and(values[0], values[1])
            elif node.op == "Or":
                result = np.logical_or(values[0], values[1])
            elif node.op == "Xor":
                result = np.logical_xor(values[0], values[1])
            elif node.op == "Not":
                result = np.logical_not(values[0])
            elif node.op == "Cast":
                result = values[0]
            elif node.op == "PRelu":
                result = np.where(values[0] >= 0, values[0], values[0] * values[1])
            elif node.op == "Sum":
                result = sum(values[1:], start=values[0])
            elif node.op == "Mean":
                result = sum(values[1:], start=values[0]) / len(values)
            elif node.op == "IsNaN":
                result = np.isnan(values[0])
            elif node.op == "IsInf":
                result = np.isinf(values[0])
                if not int(node.attributes.get("detect_negative", 1)):
                    result &= ~np.isneginf(values[0])
                if not int(node.attributes.get("detect_positive", 1)):
                    result &= ~np.isposinf(values[0])
            elif node.op == "Gather":
                result = np.take(values[0], node.attributes["indices"], axis=int(node.attributes.get("axis", 0)))
            elif node.op == "Where":
                result = np.where(values[0], values[1], values[2])
            elif node.op in UNARY:
                from .interpreter import _unary
                result = _unary(node.op, values[0], node.attributes)
            elif node.op == "Min":
                result = np.minimum(values[0], values[1])
            elif node.op == "Max":
                result = np.maximum(values[0], values[1])
            elif node.op == "Pow":
                result = np.power(values[0], values[1])
            elif node.op == "Clip":
                minimum = values[1].item() if len(values) > 1 else node.attributes.get("min")
                maximum = values[2].item() if len(values) > 2 else node.attributes.get("max")
                result = np.clip(values[0], minimum, maximum)
            elif node.op == "Reshape":
                result = np.reshape(values[0], tuple(int(value) for value in values[1].flat))
            elif node.op == "Flatten":
                axis = int(node.attributes.get("axis", 1))
                leading = math.prod(values[0].shape[:axis])
                result = np.reshape(values[0], (leading, -1))
            elif node.op == "Transpose":
                perm = node.attributes.get("perm")
                result = np.transpose(values[0], axes=perm)
            elif node.op == "Concat":
                result = np.concatenate([value.reshape(-1) for value in values])
            elif node.op == "MatMul":
                result = np.matmul(values[0], values[1])
            else:
                retained.append(node)
                continue
        except ValueError as exc:
            raise CompilerError(f"constant folding failed for {node.source_name or node.op}: {exc}") from exc
        output_id = node.outputs[0]
        output = graph.tensors[output_id]
        result_dtype = {
            DType.BOOL: np.bool_, DType.FLOAT32: np.float32, DType.FLOAT64: np.float64,
            DType.INT32: np.int32, DType.INT64: np.int64,
        }[output.dtype]
        result = np.asarray(result, dtype=result_dtype)
        constant_name = f"__folded_{output_id}_{output.name}"
        graph.constants[constant_name] = ConstantTensor(
            constant_name, tuple(int(dim) for dim in result.shape), output.dtype, result.copy(), "folded", False
        )
        output.is_constant = True
        output.constant_name = constant_name
        output.shape = tuple(int(dim) for dim in result.shape)
        changed = True
    graph.nodes = retained
    if changed:
        graph.renumber_nodes()
    return changed


def eliminate_identity(graph: Graph) -> bool:
    identities = [node for node in graph.nodes if node.op == "Identity"]
    if not identities:
        return False
    for node in identities:
        _replace_tensor(graph, node.outputs[0], node.inputs[0])
    graph.nodes = [node for node in graph.nodes if node.op != "Identity"]
    graph.renumber_nodes()
    return True


def fold_layout_operations(graph: Graph) -> bool:
    """Remove static layout scaffolding while preserving feature-major order."""
    # Validate reshape semantics before replacing any neighboring layout tensor.
    # Otherwise folding a preceding Transpose would hide whether batch was first
    # or last in the original ONNX operation.
    for node in graph.nodes:
        if node.op not in {"Flatten", "Reshape", "Squeeze", "Unsqueeze"}:
            continue
        input_tensor = graph.tensors[node.inputs[0]]
        output_tensor = graph.tensors[node.outputs[0]]
        input_batch_axes = [
            axis for axis, dim in enumerate(input_tensor.shape) if not isinstance(dim, int)
        ]
        output_batch_axes = [
            axis for axis, dim in enumerate(output_tensor.shape) if not isinstance(dim, int)
        ]
        if len(input_batch_axes) != 1 or len(output_batch_axes) != 1:
            raise CompilerError(
                f"{node.op} {node.source_name!r} must preserve exactly one dynamic batch dimension"
            )

        def batch_side(shape: tuple[object, ...], axis: int) -> str:
            if axis == 0:
                return "first"
            if axis + 1 == len(shape):
                return "last"
            return "middle"

        input_side = batch_side(input_tensor.shape, input_batch_axes[0])
        output_side = batch_side(output_tensor.shape, output_batch_axes[0])
        if input_side == "middle" or output_side == "middle" or input_side != output_side:
            raise CompilerError(
                f"{node.op} {node.source_name!r} changes batch-relative element order from "
                f"{input_tensor.shape} to {output_tensor.shape}; only reshapes that keep batch first or "
                "keep batch last are compile-time no-ops"
            )

    changed = False
    retained: list[Node] = []
    for node in graph.nodes:
        if node.op not in {"Flatten", "Reshape", "Squeeze", "Transpose", "Unsqueeze"}:
            retained.append(node)
            continue
        input_id = node.inputs[0]
        output_id = node.outputs[0]
        input_tensor = graph.tensors[input_id]
        output_tensor = graph.tensors[output_id]
        # Per-sample storage is flat. Moving only the batch axis, flattening, or statically reshaping the same number
        # of sample elements therefore changes type/shape metadata but not the generated element order.
        if input_tensor.sample_size != output_tensor.sample_size:
            raise CompilerError(
                f"{node.op} {node.source_name!r} changes per-sample element count from "
                f"{input_tensor.sample_size} to {output_tensor.sample_size}"
            )
        if node.op == "Transpose" and len(input_tensor.sample_shape) > 1:
            raise CompilerError(
                f"runtime Transpose {node.source_name!r} changes non-batch axes; only batch-axis movement and "
                "constant weight transposes are supported"
            )
        _replace_tensor(graph, output_id, input_id)
        changed = True
    graph.nodes = retained
    if changed:
        graph.renumber_nodes()
    return changed


def eliminate_dead_code(graph: Graph) -> bool:
    graph.rebuild_links()
    live_tensors = set(graph.inputs) | set(graph.outputs)
    live_nodes: set[int] = set()
    pending = list(graph.outputs)
    while pending:
        tensor_id = pending.pop()
        producer = graph.tensors[tensor_id].producer
        if producer is None or producer in live_nodes:
            continue
        live_nodes.add(producer)
        node = graph.node_by_id(producer)
        for input_id in node.inputs:
            if input_id not in live_tensors:
                live_tensors.add(input_id)
                pending.append(input_id)
    changed = len(live_nodes) != len(graph.nodes) or len(live_tensors) != len(graph.tensors)
    if not changed:
        return False
    graph.nodes = [node for node in graph.nodes if node.id in live_nodes]
    graph.tensors = {tensor_id: tensor for tensor_id, tensor in graph.tensors.items() if tensor_id in live_tensors}
    used_constants = {
        tensor.constant_name for tensor in graph.tensors.values() if tensor.is_constant and tensor.constant_name is not None
    }
    graph.constants = {name: constant for name, constant in graph.constants.items() if name in used_constants}
    graph.renumber_nodes()
    return True


def _add_constant(graph: Graph, name: str, values: np.ndarray, template_tensor_id: int, layout: str) -> int:
    new_id = max(graph.tensors, default=-1) + 1
    template = graph.tensors[template_tensor_id]
    graph.constants[name] = ConstantTensor(
        name, tuple(int(dim) for dim in values.shape), template.dtype, values.copy(), layout,
        graph.constants[template.constant_name].learned if template.constant_name is not None else False,
    )
    from .ir import TensorValue

    graph.tensors[new_id] = TensorValue(
        new_id,
        name,
        tuple(int(dim) for dim in values.shape),
        template.dtype,
        is_constant=True,
        constant_name=name,
    )
    return new_id


def canonicalize_dense(graph: Graph) -> bool:
    """Convert MatMul/Gemm spellings into PONNI's common Dense operation."""
    changed = False
    for node in graph.nodes:
        if node.op == "Gemm":
            if len(node.inputs) < 2:
                raise CompilerError(f"Gemm {node.source_name!r} requires data and weight inputs")
            data_id, weight_id = node.inputs[:2]
            weight = _constant(graph, weight_id)
            if weight is None or weight.ndim != 2:
                raise CompilerError(f"Gemm {node.source_name!r} requires a constant rank-two weight")
            trans_a = int(node.attributes.get("transA", 0))
            trans_b = int(node.attributes.get("transB", 0))
            alpha = float(node.attributes.get("alpha", 1.0))
            beta = float(node.attributes.get("beta", 1.0))
            if trans_a != 0:
                raise CompilerError(f"Gemm {node.source_name!r} uses transA=1, which is unsupported for sample data")
            effective = weight.T if trans_b else weight
            canonical_weight = np.asarray(alpha * effective.T, dtype=weight.dtype)
            weight_name = f"dense_{node.id}_weight"
            canonical_weight_id = _add_constant(graph, weight_name, canonical_weight, weight_id, "output_input")
            bias_id = None
            if len(node.inputs) >= 3:
                bias = _constant(graph, node.inputs[2])
                if bias is None or bias.size not in (1, canonical_weight.shape[0]):
                    raise CompilerError(f"Gemm {node.source_name!r} has unsupported non-vector bias broadcasting")
                canonical_bias = np.asarray(beta * bias.reshape(-1), dtype=weight.dtype)
                bias_id = _add_constant(graph, f"dense_{node.id}_bias", canonical_bias, node.inputs[2], "output")
            node.op = "Dense"
            node.inputs = [data_id, canonical_weight_id]
            if bias_id is not None:
                node.inputs.append(bias_id)
            node.attributes = {"weight": canonical_weight_id, "bias": bias_id}
            changed = True
        elif node.op == "MatMul":
            if len(node.inputs) != 2:
                raise CompilerError(f"MatMul {node.source_name!r} requires exactly two inputs")
            data_id, weight_id = node.inputs
            weight = _constant(graph, weight_id)
            if weight is None or weight.ndim != 2:
                raise CompilerError(
                    f"MatMul {node.source_name!r} requires its right operand to be a constant rank-two weight"
                )
            canonical_weight = np.asarray(weight.T, dtype=weight.dtype)
            canonical_weight_id = _add_constant(
                graph, f"dense_{node.id}_weight", canonical_weight, weight_id, "output_input"
            )
            node.op = "Dense"
            node.inputs = [data_id, canonical_weight_id]
            node.attributes = {"weight": canonical_weight_id, "bias": None}
            changed = True
    if changed:
        graph.rebuild_links()
    return changed


def fuse_dense_bias(graph: Graph) -> bool:
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.op != "Dense" or node.attributes.get("bias") is not None:
            continue
        output_id = node.outputs[0]
        consumers = graph.tensors[output_id].consumers
        if len(consumers) != 1:
            continue
        add = graph.node_by_id(consumers[0])
        if add.op != "Add":
            continue
        other_ids = [tensor_id for tensor_id in add.inputs if tensor_id != output_id]
        if len(other_ids) != 1:
            continue
        bias = _constant(graph, other_ids[0])
        out_size = graph.tensors[add.outputs[0]].sample_size
        if bias is None or bias.size not in (1, out_size):
            continue
        node.attributes["bias"] = other_ids[0]
        node.inputs.append(other_ids[0])
        node.outputs = add.outputs
        remove.add(add.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_dense_activation(graph: Graph) -> bool:
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.op != "Dense":
            continue
        output_id = node.outputs[0]
        consumers = graph.tensors[output_id].consumers
        if len(consumers) != 1:
            continue
        activation = graph.node_by_id(consumers[0])
        if activation.op not in ACTIVATIONS:
            continue
        node.op = "DenseBiasActivation"
        node.attributes["activation"] = activation.op
        node.attributes["activation_attributes"] = dict(activation.attributes)
        node.outputs = activation.outputs
        remove.add(activation.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_virtual_dense_inputs(graph: Graph) -> bool:
    """Represent static Concat/Gather chains as dense-input index maps."""
    """Let a dense read recursively composed static Concat/Gather index maps."""
    graph.rebuild_links()
    changed = False

    def index_map(tensor_id: int, active: set[int]) -> list[dict[str, int]]:
        if tensor_id in active:
            raise CompilerError("virtual dense input contains a cyclic Concat/Gather region")
        tensor = graph.tensors[tensor_id]
        if tensor.producer is None:
            return [{"tensor": tensor_id, "index": index} for index in range(tensor.sample_size)]
        producer = graph.node_by_id(tensor.producer)
        if producer.op not in {"Concat", "Gather"}:
            return [{"tensor": tensor_id, "index": index} for index in range(tensor.sample_size)]
        active.add(tensor_id)
        if producer.op == "Concat":
            result = [entry for input_id in producer.inputs for entry in index_map(input_id, active)]
        else:
            source = index_map(producer.inputs[0], active)
            result = [dict(source[int(index)]) for index in producer.attributes["indices"]]
        active.remove(tensor_id)
        if len(result) != tensor.sample_size:
            raise CompilerError(
                f"virtual {producer.op} input map has {len(result)} elements; expected {tensor.sample_size}"
            )
        return result

    for node in graph.nodes:
        if node.op != "Dense" or "input_map" in node.attributes:
            continue
        data_id = node.inputs[0]
        data = graph.tensors[data_id]
        if data.producer is None:
            continue
        producer = graph.node_by_id(data.producer)
        if producer.op not in {"Concat", "Gather"}:
            continue
        input_map = index_map(data_id, set())
        if len(input_map) != data.sample_size:
            continue
        dynamic_inputs: list[int] = []
        for entry in input_map:
            tensor_id = entry["tensor"]
            if tensor_id not in dynamic_inputs:
                dynamic_inputs.append(tensor_id)
        parameter_inputs = [int(node.attributes["weight"])]
        if node.attributes.get("bias") is not None:
            parameter_inputs.append(int(node.attributes["bias"]))
        node.inputs = dynamic_inputs + parameter_inputs
        node.attributes["input_map"] = input_map
        changed = True
    if changed:
        graph.rebuild_links()
    return changed


def prune_dense_gather_outputs(graph: Graph) -> bool:
    """Select static dense rows directly instead of materializing then gathering."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.op != "Dense":
            continue
        output = graph.tensors[node.outputs[0]]
        if len(output.consumers) != 1:
            continue
        gather = graph.node_by_id(output.consumers[0])
        if gather.op != "Gather" or gather.inputs[0] != output.id:
            continue
        indices = [int(index) for index in gather.attributes["indices"]]
        weight_id = int(node.attributes["weight"])
        weight = _constant(graph, weight_id)
        if weight is None or weight.ndim != 2:
            continue
        bias_id = node.attributes.get("bias")
        bias = None if bias_id is None else _constant(graph, int(bias_id))
        if bias_id is not None and bias is None:
            continue
        new_weight_id = _add_constant(
            graph, f"dense_{node.id}_gathered_weight", np.asarray(weight[indices, :]), weight_id, "output_input"
        )
        replacements = {weight_id: new_weight_id}
        node.attributes["weight"] = new_weight_id
        if bias_id is not None:
            assert bias is not None
            gathered_bias = bias if bias.size == 1 else np.asarray(bias.reshape(-1)[indices])
            new_bias_id = _add_constant(
                graph, f"dense_{node.id}_gathered_bias", gathered_bias, int(bias_id), "output"
            )
            replacements[int(bias_id)] = new_bias_id
            node.attributes["bias"] = new_bias_id
        node.inputs = [replacements.get(tensor_id, tensor_id) for tensor_id in node.inputs]
        node.outputs = gather.outputs
        remove.add(gather.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_residual_activation(graph: Graph) -> bool:
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.op != "Add":
            continue
        consumers = graph.tensors[node.outputs[0]].consumers
        if len(consumers) != 1:
            continue
        activation = graph.node_by_id(consumers[0])
        if activation.op not in ACTIVATIONS:
            continue
        node.op = "ResidualAddActivation"
        node.attributes["activation"] = activation.op
        node.attributes["activation_attributes"] = dict(activation.attributes)
        node.outputs = activation.outputs
        remove.add(activation.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_dense_residual_activation(graph: Graph) -> bool:
    """Fold a sole-consumer residual epilogue into the dense output loop."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.op != "Dense":
            continue
        consumers = graph.tensors[node.outputs[0]].consumers
        if len(consumers) != 1:
            continue
        residual = graph.node_by_id(consumers[0])
        if residual.op != "ResidualAddActivation":
            continue
        other = [tensor_id for tensor_id in residual.inputs if tensor_id != node.outputs[0]]
        if len(other) != 1:
            continue
        node.op = "DenseResidualActivation"
        node.inputs.append(other[0])
        node.attributes["residual"] = other[0]
        node.attributes["activation"] = residual.attributes["activation"]
        node.attributes["activation_attributes"] = dict(residual.attributes.get("activation_attributes", {}))
        node.outputs = residual.outputs
        remove.add(residual.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_dense_epilogues(graph: Graph) -> bool:
    """Attach otherwise-unhandled ordered floating-point pointwise steps to a dense loop."""
    graph.rebuild_links()
    supported = ELEMENTWISE | UNARY | {"BatchNormalization", "Clip", "Mean", "PRelu", "Sum", "Where"}
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.id in remove or node.op != "Dense":
            continue
        previous = node.outputs[0]
        steps: list[dict[str, object]] = []
        while len(set(graph.tensors[previous].consumers)) == 1:
            consumer = graph.node_by_id(graph.tensors[previous].consumers[0])
            if consumer.id in remove or consumer.op not in supported or previous not in consumer.inputs:
                break
            if graph.tensors[consumer.outputs[0]].dtype not in {DType.FLOAT32, DType.FLOAT64}:
                break
            steps.append({
                "op": consumer.op,
                "inputs": ["prev" if tensor_id == previous else tensor_id for tensor_id in consumer.inputs],
                "attributes": dict(consumer.attributes),
            })
            remove.add(consumer.id)
            previous = consumer.outputs[0]
        if not steps:
            continue
        external_inputs = list(node.inputs)
        for step in steps:
            for value in step["inputs"]:
                if value != "prev" and value not in external_inputs:
                    external_inputs.append(int(value))
        node.op = "DenseEpilogue"
        node.inputs = external_inputs
        node.outputs = [previous]
        node.attributes["epilogue_steps"] = steps
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_silu(graph: Graph) -> bool:
    """Recognize the standard ONNX Sigmoid(x) * x spelling without duplicating x reads."""
    graph.rebuild_links()
    remove: set[int] = set()
    changed = False
    for sigmoid in graph.nodes:
        if sigmoid.op != "Sigmoid" or len(graph.tensors[sigmoid.outputs[0]].consumers) != 1:
            continue
        multiply = graph.node_by_id(graph.tensors[sigmoid.outputs[0]].consumers[0])
        if multiply.op != "Mul" or len(multiply.inputs) != 2:
            continue
        if sorted(multiply.inputs) != sorted([sigmoid.inputs[0], sigmoid.outputs[0]]):
            continue
        multiply.op = "Silu"
        multiply.inputs = [sigmoid.inputs[0]]
        multiply.attributes = {}
        remove.add(sigmoid.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def canonicalize_decomposed_activations(graph: Graph) -> bool:
    """Recognize exact exporter decompositions of native activations."""
    """Recognize common activation spellings emitted as small pointwise graphs."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for multiply in graph.nodes:
        if multiply.id in remove or multiply.op != "Mul" or len(multiply.inputs) != 2:
            continue
        for value_id, branch_id in (multiply.inputs, reversed(multiply.inputs)):
            branch = graph.tensors[branch_id]
            if branch.producer is None or branch.consumers != [multiply.id]:
                continue
            outer = graph.node_by_id(branch.producer)
            if outer.op == "HardSigmoid" and outer.inputs == [value_id]:
                multiply.op = "HardSwish"
                multiply.inputs = [value_id]
                multiply.attributes = {}
                remove.add(outer.id)
                changed = True
                break
            if outer.op != "Tanh" or len(outer.inputs) != 1:
                continue
            softplus_value = graph.tensors[outer.inputs[0]]
            if softplus_value.producer is None or softplus_value.consumers != [outer.id]:
                continue
            softplus = graph.node_by_id(softplus_value.producer)
            if softplus.op != "Softplus" or softplus.inputs != [value_id]:
                continue
            multiply.op = "Mish"
            multiply.inputs = [value_id]
            multiply.attributes = {}
            remove.update({outer.id, softplus.id})
            changed = True
            break
    for final in graph.nodes:
        if final.id in remove or final.op != "Mul" or len(final.inputs) != 2:
            continue
        half_matches = [
            (tensor_id, other_id)
            for tensor_id, other_id in (final.inputs, reversed(final.inputs))
            if (value := _constant(graph, tensor_id)) is not None and value.size == 1 and
            np.isclose(float(value.item()), 0.5)
        ]
        if not half_matches:
            continue
        _, product_id = half_matches[0]
        product_tensor = graph.tensors[product_id]
        if product_tensor.producer is None or product_tensor.consumers != [final.id]:
            continue
        product = graph.node_by_id(product_tensor.producer)
        if product.op != "Mul":
            continue
        matched_nodes: tuple[Node, Node, Node] | None = None
        value_id: int | None = None
        for possible_value, add_id in (product.inputs, reversed(product.inputs)):
            add_tensor = graph.tensors[add_id]
            if add_tensor.producer is None or add_tensor.consumers != [product.id]:
                continue
            addition = graph.node_by_id(add_tensor.producer)
            if addition.op != "Add":
                continue
            for one_id, erf_id in (addition.inputs, reversed(addition.inputs)):
                one = _constant(graph, one_id)
                erf_tensor = graph.tensors[erf_id]
                if one is None or one.size != 1 or not np.isclose(float(one.item()), 1.0):
                    continue
                if erf_tensor.producer is None or erf_tensor.consumers != [addition.id]:
                    continue
                error_function = graph.node_by_id(erf_tensor.producer)
                if error_function.op != "Erf":
                    continue
                scaled_tensor = graph.tensors[error_function.inputs[0]]
                if scaled_tensor.producer is None or scaled_tensor.consumers != [error_function.id]:
                    continue
                scaled = graph.node_by_id(scaled_tensor.producer)
                if scaled.op not in {"Div", "Mul"}:
                    continue
                if scaled.op == "Div":
                    scale = _constant(graph, scaled.inputs[1])
                    legal_scale = scale is not None and scale.size == 1 and np.isclose(
                        float(scale.item()), math.sqrt(2.0), rtol=1e-5,
                    )
                    scaled_value = scaled.inputs[0]
                else:
                    scale = _constant(graph, scaled.inputs[1])
                    legal_scale = scale is not None and scale.size == 1 and np.isclose(
                        float(scale.item()), 1.0 / math.sqrt(2.0), rtol=1e-5,
                    )
                    scaled_value = scaled.inputs[0]
                if legal_scale and possible_value == scaled_value:
                    value_id = possible_value
                    matched_nodes = addition, error_function, scaled
                    break
            if matched_nodes is not None:
                break
        if matched_nodes is None or value_id is None:
            continue
        final.op = "Gelu"
        final.inputs = [value_id]
        final.attributes = {"approximate": "none"}
        remove.update({product.id, *(node.id for node in matched_nodes)})
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def canonicalize_decomposed_softmax(graph: Graph) -> bool:
    """Replace a reviewed stable Softmax/LogSoftmax decomposition."""
    """Recognize stable full-feature Softmax and LogSoftmax reduction DAGs."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()

    def producer(tensor_id: int, op: str) -> Node | None:
        producer_id = graph.tensors[tensor_id].producer
        if producer_id is None:
            return None
        node = graph.node_by_id(producer_id)
        return node if node.op == op and node.id not in remove else None

    for final in graph.nodes:
        if final.id in remove or final.op not in {"Div", "Sub"}:
            continue
        shifted_id: int | None = None
        exponential: Node | None = None
        reduction: Node | None = None
        logarithm: Node | None = None
        if final.op == "Div":
            exponential = producer(final.inputs[0], "Exp")
            reduction = producer(final.inputs[1], "ReduceSum")
            if exponential is None or reduction is None or reduction.inputs[0] != exponential.outputs[0]:
                continue
            shifted_id = exponential.inputs[0]
            canonical_op = "Softmax"
        else:
            logarithm = producer(final.inputs[1], "Log")
            if logarithm is None:
                continue
            reduction = producer(logarithm.inputs[0], "ReduceSum")
            if reduction is None:
                continue
            exponential = producer(reduction.inputs[0], "Exp")
            if exponential is None or final.inputs[0] != exponential.inputs[0]:
                continue
            shifted_id = final.inputs[0]
            canonical_op = "LogSoftmax"
        shifted = producer(shifted_id, "Sub")
        if shifted is None:
            continue
        maximum = producer(shifted.inputs[1], "ReduceMax")
        if maximum is None or maximum.inputs[0] != shifted.inputs[0]:
            continue
        expected_exp_consumers = {reduction.id, final.id} if canonical_op == "Softmax" else {reduction.id}
        if (graph.tensors[maximum.outputs[0]].consumers != [shifted.id] or
                graph.tensors[shifted.outputs[0]].consumers != [exponential.id] and
                set(graph.tensors[shifted.outputs[0]].consumers) != {exponential.id, final.id} or
                set(graph.tensors[exponential.outputs[0]].consumers) != expected_exp_consumers or
                graph.tensors[reduction.outputs[0]].consumers !=
                ([final.id] if logarithm is None else [logarithm.id]) or
                logarithm is not None and graph.tensors[logarithm.outputs[0]].consumers != [final.id]):
            continue
        final.op = canonical_op
        final.inputs = [shifted.inputs[0]]
        final.attributes = {"axis": -1}
        remove.update({maximum.id, shifted.id, exponential.id, reduction.id})
        if logarithm is not None:
            remove.add(logarithm.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def canonicalize_decomposed_layernorm(graph: Graph) -> bool:
    """Replace the reviewed reduction-based LayerNormalization decomposition."""
    """Recognize the conventional mean/variance LayerNormalization DAG."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()

    def producer(tensor_id: int, op: str | set[str]) -> Node | None:
        producer_id = graph.tensors[tensor_id].producer
        if producer_id is None:
            return None
        node = graph.node_by_id(producer_id)
        allowed = {op} if isinstance(op, str) else op
        return node if node.op in allowed and node.id not in remove else None

    for final in graph.nodes:
        if final.id in remove or final.op not in {"Add", "Mul"}:
            continue
        if final.op == "Mul" and len(graph.tensors[final.outputs[0]].consumers) == 1:
            possible_bias = graph.node_by_id(graph.tensors[final.outputs[0]].consumers[0])
            if possible_bias.op == "Add":
                other = [tensor_id for tensor_id in possible_bias.inputs if tensor_id != final.outputs[0]]
                if len(other) == 1 and _constant(graph, other[0]) is not None:
                    continue
        bias_id: int | None = None
        scaled = final
        if final.op == "Add":
            candidates = [(producer(final.inputs[0], "Mul"), final.inputs[1]),
                          (producer(final.inputs[1], "Mul"), final.inputs[0])]
            match = next(
                ((node, bias) for node, bias in candidates
                 if node is not None and _constant(graph, bias) is not None),
                None,
            )
            if match is None:
                continue
            scaled, bias_id = match
        norm_candidates = [(producer(scaled.inputs[0], "Div"), scaled.inputs[1]),
                           (producer(scaled.inputs[1], "Div"), scaled.inputs[0])]
        norm_match = next(
            ((node, scale) for node, scale in norm_candidates if node is not None and _constant(graph, scale) is not None),
            None,
        )
        if norm_match is None:
            continue
        normalized, scale_id = norm_match
        centered = producer(normalized.inputs[0], "Sub")
        root = producer(normalized.inputs[1], "Sqrt")
        if centered is None or root is None:
            continue
        variance_epsilon = producer(root.inputs[0], "Add")
        if variance_epsilon is None:
            continue
        variance_candidates = [
            (producer(variance_epsilon.inputs[0], "ReduceMean"), variance_epsilon.inputs[1]),
            (producer(variance_epsilon.inputs[1], "ReduceMean"), variance_epsilon.inputs[0]),
        ]
        variance_match = next(
            ((node, epsilon) for node, epsilon in variance_candidates
             if node is not None and _constant(graph, epsilon) is not None),
            None,
        )
        if variance_match is None:
            continue
        variance, epsilon_id = variance_match
        epsilon_values = _constant(graph, epsilon_id)
        if epsilon_values is None or epsilon_values.size != 1:
            continue
        square = producer(variance.inputs[0], {"Mul", "Pow"})
        if square is None:
            continue
        if square.op == "Mul":
            if square.inputs != [centered.outputs[0], centered.outputs[0]]:
                continue
        else:
            exponent = _constant(graph, square.inputs[1])
            if square.inputs[0] != centered.outputs[0] or exponent is None or exponent.size != 1 or exponent.item() != 2:
                continue
        mean = producer(centered.inputs[1], "ReduceMean")
        if mean is None or mean.inputs[0] != centered.inputs[0]:
            continue
        expected_centered_consumers = {square.id, normalized.id}
        if (set(graph.tensors[centered.outputs[0]].consumers) != expected_centered_consumers or
                graph.tensors[mean.outputs[0]].consumers != [centered.id] or
                graph.tensors[square.outputs[0]].consumers != [variance.id] or
                graph.tensors[variance.outputs[0]].consumers != [variance_epsilon.id] or
                graph.tensors[variance_epsilon.outputs[0]].consumers != [root.id] or
                graph.tensors[root.outputs[0]].consumers != [normalized.id] or
                graph.tensors[normalized.outputs[0]].consumers != [scaled.id] or
                scaled.id != final.id and graph.tensors[scaled.outputs[0]].consumers != [final.id]):
            continue
        final.op = "LayerNormalization"
        final.inputs = [centered.inputs[0], scale_id] + ([] if bias_id is None else [bias_id])
        final.attributes = {"axis": -1, "epsilon": float(epsilon_values.item()), "stash_type": 1}
        remove.update({mean.id, centered.id, square.id, variance.id, variance_epsilon.id, root.id, normalized.id})
        if scaled.id != final.id:
            remove.add(scaled.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_elementwise_chains(graph: Graph) -> bool:
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.id in remove or node.op not in ELEMENTWISE | UNARY:
            continue
        steps = [{"op": node.op, "inputs": list(node.inputs), "attributes": dict(node.attributes)}]
        final_output = node.outputs[0]
        while len(set(graph.tensors[final_output].consumers)) == 1:
            consumer = graph.node_by_id(graph.tensors[final_output].consumers[0])
            if consumer.id in remove or consumer.op not in ELEMENTWISE | UNARY:
                break
            if final_output not in consumer.inputs:
                break
            steps.append(
                {
                    "op": consumer.op,
                    "inputs": ["prev" if tensor_id == final_output else tensor_id for tensor_id in consumer.inputs],
                    "attributes": dict(consumer.attributes),
                }
            )
            remove.add(consumer.id)
            final_output = consumer.outputs[0]
        if len(steps) > 1:
            node.op = "ElementwiseChain"
            node.attributes = {"steps": steps}
            external_inputs: list[int] = []
            for step in steps:
                for value in step["inputs"]:
                    if value != "prev" and value not in external_inputs:
                        external_inputs.append(value)
            node.inputs = external_inputs
            node.outputs = [final_output]
            changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_mapped_reductions(graph: Graph) -> bool:
    """Evaluate a sole-consumer pointwise chain directly inside a one-pass reduction."""
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    supported = {
        "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd",
        "ReduceSum", "ReduceSumSquare",
    }
    for reduction in graph.nodes:
        if reduction.op not in supported:
            continue
        mapped_id = reduction.inputs[0]
        mapped = graph.tensors[mapped_id]
        if mapped.producer is None or mapped.consumers != [reduction.id]:
            continue
        producer = graph.node_by_id(mapped.producer)
        if producer.op == "ElementwiseChain":
            steps = producer.attributes["steps"]
            reduction.attributes["map_steps"] = steps
        elif producer.op == "PointwiseRegion":
            reduction.attributes["map_region_steps"] = producer.attributes["steps"]
            reduction.attributes["map_output"] = producer.outputs[0]
        elif producer.op in ELEMENTWISE | UNARY:
            steps = [{"op": producer.op, "inputs": list(producer.inputs),
                      "attributes": dict(producer.attributes)}]
            reduction.attributes["map_steps"] = steps
        else:
            continue
        reduction.attributes["map_size"] = mapped.sample_size
        reduction.inputs = list(producer.inputs)
        remove.add(producer.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


def fuse_pointwise_regions(graph: Graph) -> bool:
    """Fuse connected pointwise DAGs without changing expression ordering."""
    """Fuse reconverging pointwise producers into one per-element loop."""
    graph.rebuild_links()
    supported = POINTWISE | {"ElementwiseChain"}
    removed: set[int] = set()
    changed = False
    for sink in reversed(graph.nodes):
        if sink.id in removed or sink.op not in supported:
            continue
        region = {sink.id}
        grew = True
        while grew:
            grew = False
            for node_id in tuple(region):
                node = graph.node_by_id(node_id)
                for tensor_id in node.inputs:
                    producer_id = graph.tensors[tensor_id].producer
                    if producer_id is None or producer_id in region or producer_id in removed:
                        continue
                    producer = graph.node_by_id(producer_id)
                    if producer.op not in supported:
                        continue
                    if set(graph.tensors[tensor_id].consumers) <= region:
                        region.add(producer_id)
                        grew = True
        if len(region) == 1:
            continue
        pending = {node.id: node for node in graph.nodes if node.id in region}
        region_nodes: list[Node] = []
        emitted: set[int] = set()
        while pending:
            ready = sorted(
                node_id for node_id, node in pending.items()
                if all(graph.tensors[tensor_id].producer not in region or
                       graph.tensors[tensor_id].producer in emitted for tensor_id in node.inputs)
            )
            if not ready:
                raise CompilerError("pointwise fusion encountered a cyclic region")
            for node_id in ready:
                region_nodes.append(pending.pop(node_id))
                emitted.add(node_id)
        external_inputs: list[int] = []
        for node in region_nodes:
            for tensor_id in node.inputs:
                if graph.tensors[tensor_id].producer not in region and tensor_id not in external_inputs:
                    external_inputs.append(tensor_id)
        steps = [
            {"id": node.id, "op": node.op, "inputs": list(node.inputs), "outputs": list(node.outputs),
             "attributes": dict(node.attributes), "output_dtype": graph.tensors[node.outputs[0]].dtype.value,
             "input_dtypes": [graph.tensors[tensor_id].dtype.value for tensor_id in node.inputs]}
            for node in region_nodes
        ]
        sink.op = "PointwiseRegion"
        sink.inputs = external_inputs
        sink.attributes = {"steps": steps}
        removed.update(region - {sink.id})
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in removed]
        graph.renumber_nodes()
    return changed


def fuse_comparison_where(graph: Graph) -> bool:
    """Fuse a sole-consumer comparison mask directly into floating-point selection."""
    graph.rebuild_links()
    remove: set[int] = set()
    changed = False
    for node in graph.nodes:
        if node.op != "Where":
            continue
        condition = graph.tensors[node.inputs[0]]
        if condition.producer is None or condition.consumers != [node.id]:
            continue
        comparison = graph.node_by_id(condition.producer)
        if comparison.op not in COMPARISONS:
            continue
        if graph.tensors[comparison.inputs[0]].dtype == DType.BOOL:
            continue
        node.op = "CompareSelect"
        node.inputs = comparison.inputs + node.inputs[1:]
        node.attributes = {"comparison": comparison.op}
        remove.add(comparison.id)
        changed = True
    if changed:
        graph.nodes = [node for node in graph.nodes if node.id not in remove]
        graph.renumber_nodes()
    return changed


PASS_STAGES: list[list[tuple[str, Callable[[Graph], bool]]]] = [
    [
        ("topological-schedule", topological_schedule),
        ("constant-fold", constant_fold),
        ("identity-elimination", eliminate_identity),
        ("layout-fold", fold_layout_operations),
        ("dead-code-elimination", eliminate_dead_code),
    ],
    [
        ("dense-canonicalization", canonicalize_dense),
        ("dense-gather-pruning", prune_dense_gather_outputs),
        ("dense-bias-fusion", fuse_dense_bias),
        ("virtual-dense-input-fusion", fuse_virtual_dense_inputs),
    ],
    [
        ("silu-fusion", fuse_silu),
        ("decomposed-activation-canonicalization", canonicalize_decomposed_activations),
        ("decomposed-softmax-canonicalization", canonicalize_decomposed_softmax),
        ("decomposed-layernorm-canonicalization", canonicalize_decomposed_layernorm),
    ],
    [
        ("dense-activation-fusion", fuse_dense_activation),
        ("residual-activation-fusion", fuse_residual_activation),
        ("dense-residual-activation-fusion", fuse_dense_residual_activation),
        ("dense-epilogue-fusion", fuse_dense_epilogues),
    ],
    [
        ("comparison-where-fusion", fuse_comparison_where),
        ("elementwise-chain-fusion", fuse_elementwise_chains),
        ("pointwise-region-fusion", fuse_pointwise_regions),
        ("mapped-reduction-fusion", fuse_mapped_reductions),
    ],
    [
        ("dead-code-cleanup", eliminate_dead_code),
        ("final-schedule", topological_schedule),
    ],
]
# Stage boundaries explain intentional repetition: fusion can expose new
# cleanup opportunities before the next recognizer family runs.
PASS_PIPELINE: list[tuple[str, Callable[[Graph], bool]]] = [entry for stage in PASS_STAGES for entry in stage]


def optimize(graph: Graph, disabled: set[str] | None = None) -> tuple[Graph, list[dict[str, object]]]:
    """Run the ordered pipeline and return an auditable per-pass report."""
    disabled = disabled or set()
    unknown = disabled - {name for name, _ in PASS_PIPELINE}
    if unknown:
        raise CompilerError(f"unknown disabled optimization pass(es): {', '.join(sorted(unknown))}")
    optimized = graph.clone()
    report: list[dict[str, object]] = []
    for stage in PASS_STAGES:
        entries = {
            name: {"name": name, "disabled": name in disabled, "changed": False,
                   "nodes_before": len(optimized.nodes), "nodes_after": len(optimized.nodes), "iterations": 0}
            for name, _ in stage
        }
        for iteration in range(8):
            stage_changed = False
            for name, function in stage:
                entry = entries[name]
                if name in disabled:
                    continue
                changed = function(optimized)
                entry["changed"] = bool(entry["changed"] or changed)
                entry["nodes_after"] = len(optimized.nodes)
                entry["iterations"] = iteration + 1
                stage_changed = stage_changed or changed
            if not stage_changed:
                break
        else:
            names = ", ".join(name for name, _ in stage)
            raise CompilerError(f"optimization stage did not converge after eight iterations: {names}")
        report.extend(entries[name] for name, _ in stage)
    return optimized, report
