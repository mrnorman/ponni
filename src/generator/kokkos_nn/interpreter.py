from __future__ import annotations

import numpy as np

from .errors import CompilerError
from .ir import Graph


def _unary(name: str, value: np.ndarray, attributes: dict[str, object] | None = None) -> np.ndarray:
    attributes = attributes or {}
    if name == "Relu":
        return np.maximum(value, 0)
    if name == "Sigmoid":
        positive = value >= 0
        result = np.empty_like(value)
        result[positive] = 1 / (1 + np.exp(-value[positive]))
        exp_value = np.exp(value[~positive])
        result[~positive] = exp_value / (1 + exp_value)
        return result
    if name == "Tanh":
        return np.tanh(value)
    if name == "LeakyRelu":
        alpha = float(attributes.get("alpha", 0.01))
        return np.where(value >= 0, value, alpha * value)
    if name == "Elu":
        alpha = float(attributes.get("alpha", 1.0))
        return np.where(value >= 0, value, alpha * np.expm1(value))
    if name == "Gelu":
        if attributes.get("approximate", "none") == "tanh":
            return 0.5 * value * (1 + np.tanh(np.sqrt(2 / np.pi) * (value + 0.044715 * value**3)))
        import math
        return 0.5 * value * (1 + np.vectorize(math.erf)(value / np.sqrt(2)))
    if name == "Silu":
        return value / (1 + np.exp(-value))
    if name == "Softplus":
        return np.logaddexp(value, 0)
    if name == "HardSigmoid":
        return np.clip(float(attributes.get("alpha", 0.2)) * value + float(attributes.get("beta", 0.5)), 0, 1)
    if name == "HardSwish":
        return value * np.clip(value / 6 + 0.5, 0, 1)
    if name == "Mish":
        return value * np.tanh(np.logaddexp(value, 0))
    if name == "Abs":
        return np.abs(value)
    if name == "Neg":
        return -value
    if name == "Exp":
        return np.exp(value)
    if name == "Log":
        return np.log(value)
    if name == "Sqrt":
        return np.sqrt(value)
    if name == "Reciprocal":
        return np.reciprocal(value)
    raise CompilerError(f"interpreter has no unary implementation for {name}")


def _binary(op: str, left: np.ndarray, right: np.ndarray, output_size: int) -> np.ndarray:
    if left.size not in (1, output_size) or right.size not in (1, output_size):
        raise CompilerError(
            f"unsupported {op} broadcasting: operand sizes {left.size} and {right.size}, output size {output_size}"
        )
    if op == "Add":
        return left + right
    if op == "Sub":
        return left - right
    if op == "Mul":
        return left * right
    if op == "Div":
        return left / right
    if op == "Min":
        return np.minimum(left, right)
    if op == "Max":
        return np.maximum(left, right)
    if op == "Pow":
        return np.power(left, right)
    raise CompilerError(f"interpreter has no binary implementation for {op}")


def run_sample(graph: Graph, sample: np.ndarray) -> np.ndarray:
    values: dict[int, np.ndarray] = {}
    input_id = graph.inputs[0]
    expected = graph.tensors[input_id].sample_size
    flat_sample = np.asarray(sample).reshape(-1)
    if flat_sample.size != expected:
        raise CompilerError(f"IR input has {flat_sample.size} values; expected {expected}")
    values[input_id] = flat_sample
    for tensor_id, tensor in graph.tensors.items():
        if tensor.is_constant and tensor.constant_name is not None:
            values[tensor_id] = graph.constants[tensor.constant_name].values

    for node in graph.nodes:
        inputs = [values[tensor_id] for tensor_id in node.inputs]
        output_size = graph.tensors[node.outputs[0]].sample_size
        if node.op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
            result = _binary(node.op, inputs[0], inputs[1], output_size)
        elif node.op in {"Abs", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                          "Neg", "Reciprocal", "Relu", "Sigmoid", "Silu", "Softplus", "Sqrt", "Tanh"}:
            result = _unary(node.op, inputs[0], node.attributes)
        elif node.op == "Clip":
            minimum = inputs[1].item() if len(inputs) > 1 else node.attributes.get("min")
            maximum = inputs[2].item() if len(inputs) > 2 else node.attributes.get("max")
            result = np.clip(inputs[0], minimum, maximum)
        elif node.op in {"Softmax", "LogSoftmax"}:
            shifted = inputs[0] - np.max(inputs[0])
            log_sum = np.log(np.sum(np.exp(shifted)))
            result = shifted - log_sum if node.op == "LogSoftmax" else np.exp(shifted - log_sum)
        elif node.op == "LayerNormalization":
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            mean = np.mean(inputs[0])
            normalized = (inputs[0] - mean) / np.sqrt(np.mean((inputs[0] - mean) ** 2) + epsilon)
            result = normalized * inputs[1]
            if len(inputs) == 3:
                result = result + inputs[2]
        elif node.op == "BatchNormalization":
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            result = (inputs[0] - inputs[3]) / np.sqrt(inputs[4] + epsilon) * inputs[1] + inputs[2]
        elif node.op in {"ReduceMean", "ReduceSum"}:
            reduced = np.mean(inputs[0]) if node.op == "ReduceMean" else np.sum(inputs[0])
            result = np.asarray([reduced], dtype=inputs[0].dtype)
        elif node.op == "Identity":
            result = inputs[0]
        elif node.op == "Concat":
            result = np.concatenate([value.reshape(-1) for value in inputs])
        elif node.op in {"Flatten", "Reshape"}:
            result = inputs[0].reshape(-1)
        elif node.op == "Transpose":
            result = inputs[0].reshape(-1)
        elif node.op == "MatMul":
            result = np.matmul(inputs[0], inputs[1])
        elif node.op == "Gemm":
            trans_a = int(node.attributes.get("transA", 0))
            trans_b = int(node.attributes.get("transB", 0))
            if trans_a:
                raise CompilerError("sample interpreter does not support Gemm transA=1")
            weight = inputs[1].T if trans_b else inputs[1]
            result = float(node.attributes.get("alpha", 1.0)) * np.matmul(inputs[0], weight)
            if len(inputs) > 2:
                result = result + float(node.attributes.get("beta", 1.0)) * inputs[2]
        elif node.op in {"Dense", "DenseBiasActivation"}:
            weight = values[node.attributes["weight"]]
            result = np.matmul(weight, inputs[0])
            bias_id = node.attributes.get("bias")
            if bias_id is not None:
                result = result + values[bias_id]
            if node.op == "DenseBiasActivation":
                result = _unary(node.attributes["activation"], result,
                                node.attributes.get("activation_attributes", {}))
        elif node.op == "ResidualAddActivation":
            result = _unary(node.attributes["activation"], _binary("Add", inputs[0], inputs[1], output_size),
                            node.attributes.get("activation_attributes", {}))
        elif node.op == "ElementwiseChain":
            result = None
            for step in node.attributes["steps"]:
                operands = [result if tensor_id == "prev" else values[tensor_id] for tensor_id in step["inputs"]]
                result = _binary(step["op"], operands[0], operands[1], output_size)
            assert result is not None
        else:
            raise CompilerError(f"interpreter has no implementation for canonical operation {node.op}")
        result = np.asarray(result).reshape(-1)
        if result.size != output_size:
            raise CompilerError(
                f"operation {node.op} produced {result.size} per-sample values; expected {output_size}"
            )
        values[node.outputs[0]] = result
    return values[graph.outputs[0]].copy()


def run_graph(graph: Graph, inputs: np.ndarray) -> np.ndarray:
    array = np.asarray(inputs)
    expected_inputs = graph.tensors[graph.inputs[0]].sample_size
    if array.ndim != 2 or array.shape[0] != expected_inputs:
        raise CompilerError(f"logical input shape must be ({expected_inputs}, batch_size); got {array.shape}")
    output_size = graph.tensors[graph.outputs[0]].sample_size
    outputs = np.empty((output_size, array.shape[1]), dtype=array.dtype)
    for ibatch in range(array.shape[1]):
        outputs[:, ibatch] = run_sample(graph, array[:, ibatch])
    return outputs
