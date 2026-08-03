from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable

import numpy as np

from .errors import CompilerError
from .ir import ConstantTensor, Graph, Node


ACTIVATIONS = {
    "Elu", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Mish", "Relu", "Sigmoid", "Silu",
    "Softplus", "Tanh",
}
ELEMENTWISE = {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}
UNARY = ACTIVATIONS | {"Abs", "Exp", "Log", "Neg", "Reciprocal", "Sqrt"}


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
        result = np.asarray(result, dtype=values[0].dtype)
        constant_name = f"__folded_{output_id}_{output.name}"
        graph.constants[constant_name] = ConstantTensor(
            constant_name, tuple(int(dim) for dim in result.shape), output.dtype, result.copy(), "folded"
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
    # Validate reshape semantics before replacing any neighboring layout tensor.
    # Otherwise folding a preceding Transpose would hide whether batch was first
    # or last in the original ONNX operation.
    for node in graph.nodes:
        if node.op not in {"Flatten", "Reshape"}:
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
        if node.op not in {"Flatten", "Reshape", "Transpose"}:
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
        name, tuple(int(dim) for dim in values.shape), template.dtype, values.copy(), layout
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


def fuse_elementwise_chains(graph: Graph) -> bool:
    graph.rebuild_links()
    changed = False
    remove: set[int] = set()
    for node in graph.nodes:
        if node.id in remove or node.op not in ELEMENTWISE:
            continue
        steps = [{"op": node.op, "inputs": list(node.inputs)}]
        final_output = node.outputs[0]
        while len(graph.tensors[final_output].consumers) == 1:
            consumer = graph.node_by_id(graph.tensors[final_output].consumers[0])
            if consumer.id in remove or consumer.op not in ELEMENTWISE:
                break
            external = [tensor_id for tensor_id in consumer.inputs if tensor_id != final_output]
            if len(external) != 1:
                break
            steps.append(
                {
                    "op": consumer.op,
                    "inputs": ["prev" if tensor_id == final_output else tensor_id for tensor_id in consumer.inputs],
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


PASS_PIPELINE: list[tuple[str, Callable[[Graph], bool]]] = [
    ("topological-schedule", topological_schedule),
    ("constant-fold", constant_fold),
    ("identity-elimination", eliminate_identity),
    ("layout-fold", fold_layout_operations),
    ("dead-code-elimination", eliminate_dead_code),
    ("dense-canonicalization", canonicalize_dense),
    ("dense-bias-fusion", fuse_dense_bias),
    ("silu-fusion", fuse_silu),
    ("dense-activation-fusion", fuse_dense_activation),
    ("residual-activation-fusion", fuse_residual_activation),
    ("elementwise-chain-fusion", fuse_elementwise_chains),
    ("dead-code-cleanup", eliminate_dead_code),
    ("final-schedule", topological_schedule),
]


def optimize(graph: Graph, disabled: set[str] | None = None) -> tuple[Graph, list[dict[str, object]]]:
    disabled = disabled or set()
    unknown = disabled - {name for name, _ in PASS_PIPELINE}
    if unknown:
        raise CompilerError(f"unknown disabled optimization pass(es): {', '.join(sorted(unknown))}")
    optimized = graph.clone()
    report: list[dict[str, object]] = []
    for name, function in PASS_PIPELINE:
        if name in disabled:
            report.append({"name": name, "disabled": True, "changed": False})
            continue
        before = len(optimized.nodes)
        changed = function(optimized)
        report.append({"name": name, "disabled": False, "changed": changed, "nodes_before": before,
                       "nodes_after": len(optimized.nodes)})
    return optimized, report
