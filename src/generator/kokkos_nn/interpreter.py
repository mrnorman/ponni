from __future__ import annotations

import numpy as np

from .errors import CompilerError
from .ir import DType, Graph


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
    if name == "Celu":
        alpha = float(attributes.get("alpha", 1.0))
        return np.maximum(value, 0) + np.minimum(0, alpha * np.expm1(value / alpha))
    if name == "Selu":
        alpha = float(attributes.get("alpha", 1.6732631921768188))
        gamma = float(attributes.get("gamma", 1.0507010221481323))
        return gamma * np.where(value > 0, value, alpha * np.expm1(value))
    if name == "Gelu":
        if attributes.get("approximate", "none") == "tanh":
            return 0.5 * value * (1 + np.tanh(np.sqrt(2 / np.pi) * (value + 0.044715 * value**3)))
        import math
        return 0.5 * value * (1 + np.vectorize(math.erf)(value / np.sqrt(2)))
    if name == "Silu":
        return value / (1 + np.exp(-value))
    if name == "Softplus":
        return np.logaddexp(value, 0)
    if name == "Softsign":
        return value / (1 + np.abs(value))
    if name == "ThresholdedRelu":
        return np.where(value > float(attributes.get("alpha", 1.0)), value, 0)
    if name == "HardSigmoid":
        return np.clip(float(attributes.get("alpha", 0.2)) * value + float(attributes.get("beta", 0.5)), 0, 1)
    if name == "HardSwish":
        return value * np.clip(value / 6 + 0.5, 0, 1)
    if name == "Mish":
        return value * np.tanh(np.logaddexp(value, 0))
    if name == "Abs":
        return np.abs(value)
    if name == "Acos":
        return np.arccos(value)
    if name == "Acosh":
        return np.arccosh(value)
    if name == "Asin":
        return np.arcsin(value)
    if name == "Asinh":
        return np.arcsinh(value)
    if name == "Atan":
        return np.arctan(value)
    if name == "Atanh":
        return np.arctanh(value)
    if name == "Ceil":
        return np.ceil(value)
    if name == "Cos":
        return np.cos(value)
    if name == "Cosh":
        return np.cosh(value)
    if name == "Erf":
        import math
        return np.vectorize(math.erf, otypes=[value.dtype])(value)
    if name == "Neg":
        return -value
    if name == "Exp":
        return np.exp(value)
    if name == "Floor":
        return np.floor(value)
    if name == "Log":
        return np.log(value)
    if name == "Sqrt":
        return np.sqrt(value)
    if name == "Reciprocal":
        return np.reciprocal(value)
    if name == "Round":
        return np.round(value)
    if name == "Sign":
        return np.sign(value)
    if name == "Sin":
        return np.sin(value)
    if name == "Sinh":
        return np.sinh(value)
    if name == "Tan":
        return np.tan(value)
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


def _comparison(op: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if op == "Equal":
        return np.equal(left, right)
    if op == "Greater":
        return np.greater(left, right)
    if op == "GreaterOrEqual":
        return np.greater_equal(left, right)
    if op == "Less":
        return np.less(left, right)
    if op == "LessOrEqual":
        return np.less_equal(left, right)
    raise CompilerError(f"interpreter has no comparison implementation for {op}")


def _pointwise_step(op: str, operands: list[np.ndarray], attributes: dict[str, object],
                    output_size: int, output_dtype: DType) -> np.ndarray:
    if op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
        return _binary(op, operands[0], operands[1], output_size)
    if op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
        return _comparison(op, operands[0], operands[1])
    if op == "And":
        return np.logical_and(operands[0], operands[1])
    if op == "Or":
        return np.logical_or(operands[0], operands[1])
    if op == "Xor":
        return np.logical_xor(operands[0], operands[1])
    if op == "Not":
        return np.logical_not(operands[0])
    if op == "Cast":
        dtype = np.float32 if output_dtype == DType.FLOAT32 else np.float64
        return operands[0].astype(dtype)
    if op == "PRelu":
        return np.where(operands[0] >= 0, operands[0], operands[0] * operands[1])
    if op in {"Mean", "Sum"}:
        result = sum(operands[1:], start=operands[0])
        return result / len(operands) if op == "Mean" else result
    if op == "IsNaN":
        return np.isnan(operands[0])
    if op == "IsInf":
        result = np.isinf(operands[0])
        if not int(attributes.get("detect_negative", 1)):
            result &= ~np.isneginf(operands[0])
        if not int(attributes.get("detect_positive", 1)):
            result &= ~np.isposinf(operands[0])
        return result
    if op == "Where":
        return np.where(operands[0], operands[1], operands[2])
    if op == "CompareSelect":
        condition = _comparison(str(attributes["comparison"]), operands[0], operands[1])
        return np.where(condition, operands[2], operands[3])
    if op == "Clip":
        minimum = operands[1].item() if len(operands) > 1 else attributes.get("min")
        maximum = operands[2].item() if len(operands) > 2 else attributes.get("max")
        return np.clip(operands[0], minimum, maximum)
    if op == "BatchNormalization":
        epsilon = float(attributes.get("epsilon", 1e-5))
        return ((operands[0] - operands[3]) / np.sqrt(operands[4] + epsilon) * operands[1] + operands[2])
    if len(operands) == 1:
        return _unary(op, operands[0], attributes)
    raise CompilerError(f"interpreter has no pointwise-region implementation for {op}")


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
        mapped_input = None
        if "map_steps" in node.attributes:
            for step in node.attributes["map_steps"]:
                operands = [mapped_input if tensor_id == "prev" else values[tensor_id]
                            for tensor_id in step["inputs"]]
                if len(operands) == 1:
                    mapped_input = _unary(step["op"], operands[0], step.get("attributes", {}))
                else:
                    mapped_input = _binary(step["op"], operands[0], operands[1],
                                           int(node.attributes["map_size"]))
        elif "map_region_steps" in node.attributes:
            region_values = dict(values)
            map_size = int(node.attributes["map_size"])
            for step in node.attributes["map_region_steps"]:
                operands = [region_values[tensor_id] for tensor_id in step["inputs"]]
                if step["op"] == "ElementwiseChain":
                    step_result = None
                    for chain_step in step["attributes"]["steps"]:
                        chain_operands = [
                            step_result if tensor_id == "prev" else region_values[tensor_id]
                            for tensor_id in chain_step["inputs"]
                        ]
                        step_result = _pointwise_step(
                            chain_step["op"], chain_operands, chain_step.get("attributes", {}), map_size,
                            DType(step["output_dtype"]),
                        )
                else:
                    step_result = _pointwise_step(
                        step["op"], operands, step["attributes"], map_size,
                        DType(step["output_dtype"]),
                    )
                region_values[step["outputs"][0]] = step_result
            mapped_input = region_values[int(node.attributes["map_output"])]
        if node.op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
            result = _binary(node.op, inputs[0], inputs[1], output_size)
        elif node.op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
            result = _comparison(node.op, inputs[0], inputs[1])
        elif node.op == "And":
            result = np.logical_and(inputs[0], inputs[1])
        elif node.op == "Or":
            result = np.logical_or(inputs[0], inputs[1])
        elif node.op == "Xor":
            result = np.logical_xor(inputs[0], inputs[1])
        elif node.op == "Not":
            result = np.logical_not(inputs[0])
        elif node.op == "Cast":
            dtype = np.float32 if graph.tensors[node.outputs[0]].dtype == DType.FLOAT32 else np.float64
            result = inputs[0].astype(dtype)
        elif node.op == "PRelu":
            result = np.where(inputs[0] >= 0, inputs[0], inputs[0] * inputs[1])
        elif node.op in {"Mean", "Sum"}:
            result = sum(inputs[1:], start=inputs[0])
            if node.op == "Mean":
                result = result / len(inputs)
        elif node.op == "IsNaN":
            result = np.isnan(inputs[0])
        elif node.op == "IsInf":
            result = np.isinf(inputs[0])
            if not int(node.attributes.get("detect_negative", 1)):
                result &= ~np.isneginf(inputs[0])
            if not int(node.attributes.get("detect_positive", 1)):
                result &= ~np.isposinf(inputs[0])
        elif node.op == "Where":
            result = np.where(inputs[0], inputs[1], inputs[2])
        elif node.op == "CompareSelect":
            condition = _comparison(str(node.attributes["comparison"]), inputs[0], inputs[1])
            result = np.where(condition, inputs[2], inputs[3])
        elif node.op in {"Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos",
                          "Cosh", "Elu", "Erf", "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu",
                          "Log", "Mish", "Neg", "Reciprocal", "Relu", "Round", "Selu", "Sigmoid", "Sign", "Silu",
                          "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"}:
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
        elif node.op.startswith("Reduce"):
            reduction_input = inputs[0] if mapped_input is None else mapped_input
            if node.op == "ReduceL1":
                reduced = np.sum(np.abs(reduction_input))
            elif node.op == "ReduceL2":
                reduced = np.sqrt(np.sum(reduction_input ** 2))
            elif node.op == "ReduceLogSum":
                reduced = np.log(np.sum(reduction_input))
            elif node.op == "ReduceLogSumExp":
                maximum = np.max(reduction_input)
                reduced = maximum if np.isinf(maximum) else maximum + np.log(np.sum(np.exp(reduction_input - maximum)))
            elif node.op == "ReduceMax":
                reduced = np.max(reduction_input)
            elif node.op == "ReduceMean":
                reduced = np.mean(reduction_input)
            elif node.op == "ReduceMin":
                reduced = np.min(reduction_input)
            elif node.op == "ReduceProd":
                reduced = np.prod(reduction_input)
            elif node.op == "ReduceSum":
                reduced = np.sum(reduction_input)
            elif node.op == "ReduceSumSquare":
                reduced = np.sum(reduction_input ** 2)
            else:
                raise CompilerError(f"interpreter has no reduction implementation for {node.op}")
            result = np.asarray([reduced], dtype=reduction_input.dtype)
        elif node.op == "LpNormalization":
            norm = np.sum(np.abs(inputs[0])) if int(node.attributes.get("p", 2)) == 1 else np.linalg.norm(inputs[0])
            result = np.zeros_like(inputs[0]) if norm == 0 else inputs[0] / norm
        elif node.op == "Identity":
            result = inputs[0]
        elif node.op == "Concat":
            result = np.concatenate([value.reshape(-1) for value in inputs])
        elif node.op == "Gather":
            result = np.take(inputs[0], node.attributes["indices"])
        elif node.op in {"Flatten", "Reshape", "Squeeze", "Unsqueeze"}:
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
        elif node.op in {"Dense", "DenseBiasActivation", "DenseResidualActivation", "DenseEpilogue"}:
            weight = values[node.attributes["weight"]]
            if "input_map" in node.attributes:
                dense_input = np.asarray([
                    values[entry["tensor"]].reshape(-1)[entry["index"]]
                    for entry in node.attributes["input_map"]
                ])
            else:
                dense_input = inputs[0]
            result = np.matmul(weight, dense_input)
            bias_id = node.attributes.get("bias")
            if bias_id is not None:
                result = result + values[bias_id]
            if node.op == "DenseBiasActivation":
                result = _unary(node.attributes["activation"], result,
                                node.attributes.get("activation_attributes", {}))
            elif node.op == "DenseResidualActivation":
                result = _unary(
                    node.attributes["activation"], result + values[node.attributes["residual"]],
                    node.attributes.get("activation_attributes", {}),
                )
            elif node.op == "DenseEpilogue":
                for step in node.attributes["epilogue_steps"]:
                    operands = [result if value == "prev" else values[value] for value in step["inputs"]]
                    result = _pointwise_step(
                        step["op"], operands, step.get("attributes", {}), output_size,
                        graph.tensors[node.outputs[0]].dtype,
                    )
        elif node.op == "ResidualAddActivation":
            result = _unary(node.attributes["activation"], _binary("Add", inputs[0], inputs[1], output_size),
                            node.attributes.get("activation_attributes", {}))
        elif node.op == "ElementwiseChain":
            result = None
            for step in node.attributes["steps"]:
                operands = [result if tensor_id == "prev" else values[tensor_id] for tensor_id in step["inputs"]]
                if len(operands) == 1:
                    result = _unary(step["op"], operands[0], step.get("attributes", {}))
                else:
                    result = _binary(step["op"], operands[0], operands[1], output_size)
            assert result is not None
        elif node.op == "PointwiseRegion":
            region_values = dict(values)
            for step in node.attributes["steps"]:
                operands = [region_values[tensor_id] for tensor_id in step["inputs"]]
                if step["op"] == "ElementwiseChain":
                    step_result = None
                    for chain_step in step["attributes"]["steps"]:
                        chain_operands = [
                            step_result if tensor_id == "prev" else region_values[tensor_id]
                            for tensor_id in chain_step["inputs"]
                        ]
                        if len(chain_operands) == 1:
                            step_result = _unary(
                                chain_step["op"], chain_operands[0], chain_step.get("attributes", {}),
                            )
                        else:
                            step_result = _binary(
                                chain_step["op"], chain_operands[0], chain_operands[1], output_size,
                            )
                else:
                    step_result = _pointwise_step(
                        step["op"], operands, step["attributes"], output_size,
                        DType(step["output_dtype"]),
                    )
                region_values[step["outputs"][0]] = step_result
            result = region_values[node.outputs[0]]
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
