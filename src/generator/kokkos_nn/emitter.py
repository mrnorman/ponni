from __future__ import annotations

from pathlib import Path
import re

from .errors import CompilerError
from .ir import Graph, Node
from .planner import StoragePlan
from .scheduler import DenseChainSchedule


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = f"model_{identifier}"
    return identifier


def find_streaming_dense_pair(graph: Graph, maximum_output_accumulators: int = 8) -> tuple[Node, Node] | None:
    """Return the exact two-layer MLP schedule that can avoid a hidden activation array."""
    if len(graph.nodes) != 2:
        return None
    producer, consumer = graph.nodes
    if producer.op != "DenseBiasActivation" or consumer.op not in {"Dense", "DenseBiasActivation"}:
        return None
    if producer.inputs[0] != graph.inputs[0] or producer.outputs[0] != consumer.inputs[0]:
        return None
    if consumer.outputs[0] != graph.outputs[0]:
        return None
    hidden = graph.tensors[producer.outputs[0]]
    if hidden.consumers != [consumer.id]:
        return None
    if graph.tensors[consumer.outputs[0]].sample_size > maximum_output_accumulators:
        return None
    return producer, consumer


def find_streaming_dense_tail(graph: Graph, maximum_output_accumulators: int = 8) -> tuple[Node, Node] | None:
    """Return a final dense pair whose hidden value can be consumed without an array."""
    if len(graph.nodes) < 3:
        return None
    producer, consumer = graph.nodes[-2:]
    if producer.op != "DenseBiasActivation" or consumer.op not in {"Dense", "DenseBiasActivation"}:
        return None
    if producer.outputs[0] != consumer.inputs[0] or consumer.outputs[0] != graph.outputs[0]:
        return None
    if graph.tensors[producer.outputs[0]].consumers != [consumer.id]:
        return None
    if graph.tensors[consumer.outputs[0]].sample_size > maximum_output_accumulators:
        return None
    return producer, consumer


def sample_local_workspace_elements(graph: Graph, plan: StoragePlan,
                                    maximum_output_accumulators: int = 8) -> int:
    """Return the exact local extent after removing a legally streamed hidden tensor."""
    streaming_pair = find_streaming_dense_pair(graph, maximum_output_accumulators)
    if streaming_pair is not None:
        return 0
    streaming_tail = find_streaming_dense_tail(graph, maximum_output_accumulators)
    if streaming_tail is None:
        return plan.total_elements
    streamed_tensor = streaming_tail[0].outputs[0]
    return max(
        (slot.offset + slot.size for tensor_id, slot in plan.slots.items() if tensor_id != streamed_tensor),
        default=0,
    )


def half2_accumulator_heuristic(dot_length: int, simultaneous_outputs: int = 1) -> int:
    """Select a measured balanced half2 accumulator count for one dense dot product."""
    if dot_length < 2:
        return 0
    if dot_length <= 24:
        count = 2
    elif dot_length <= 80:
        count = 4
    else:
        count = 16
    # The measured width-128 streaming tail kept 16 * 3 packed partials live.
    # Do not extrapolate beyond that register-pressure point for wider outputs.
    while count > 2 and count * simultaneous_outputs > 48:
        count //= 2
    return count


def half2_accumulator_plan(graph: Graph, maximum_output_accumulators: int = 8,
                           schedule: DenseChainSchedule | None = None) -> dict[int, int]:
    """Return independent heuristic counts for canonical dense nodes."""
    result = {
        node.id: half2_accumulator_heuristic(graph.tensors[node.inputs[0]].sample_size)
        for node in graph.nodes if node.op in {"Dense", "DenseBiasActivation"}
    }
    streamed_consumers = (
        sorted(schedule.pair_by_consumer) if schedule is not None else []
    )
    if schedule is None:
        streaming = (
            find_streaming_dense_pair(graph, maximum_output_accumulators)
            or find_streaming_dense_tail(graph, maximum_output_accumulators)
        )
        if streaming is not None:
            streamed_consumers.append(streaming[1].id)
    for consumer_id in streamed_consumers:
        consumer = graph.node_by_id(consumer_id)
        result[consumer.id] = half2_accumulator_heuristic(
            graph.tensors[consumer.inputs[0]].sample_size,
            graph.tensors[consumer.outputs[0]].sample_size,
        )
    return result


def find_tensorcore_dense_chain(graph: Graph) -> tuple[Node, ...] | None:
    """Return a two- or three-dense linear chain that can be emitted as one CUDA kernel."""
    if len(graph.nodes) not in (2, 3):
        return None
    nodes = tuple(graph.nodes)
    if nodes[0].inputs[0] != graph.inputs[0] or nodes[-1].outputs[0] != graph.outputs[0]:
        return None
    for index, node in enumerate(nodes):
        required = "DenseBiasActivation" if index + 1 < len(nodes) else None
        if required is not None and node.op != required:
            return None
        if required is None and node.op not in {"Dense", "DenseBiasActivation"}:
            return None
        if index > 0 and node.inputs[0] != nodes[index - 1].outputs[0]:
            return None
        if index + 1 < len(nodes) and graph.tensors[node.outputs[0]].consumers != [nodes[index + 1].id]:
            return None
    return nodes


def estimate_tensorcore_scratch_bytes(graph: Graph, chain: tuple[Node, ...] | None) -> int:
    """Return the generated raw-CUDA dynamic shared-memory requirement per warp."""
    if chain is None or len(chain) == 2:
        return 2048
    input_size = graph.tensors[graph.inputs[0]].sample_size
    first_size = graph.tensors[chain[0].outputs[0]].sample_size
    padded_inputs = ((input_size + 7) // 8) * 8
    padded_first = ((first_size + 7) // 8) * 8
    return (16 * padded_inputs + 16 * padded_first + 16 * 8 + 16 * 16) * 4


class CppEmitter:
    def __init__(self, graph: Graph, plan: StoragePlan, sample_plan: StoragePlan,
                 schedule: DenseChainSchedule, weight_offsets: dict[int, int], model_name: str,
                 strategy: str, default_batch_tile: int, maximum_batch_tile: int) -> None:
        if strategy not in {"sample-local", "team", "tensorcore", "half2"}:
            raise CompilerError(
                f"unknown execution strategy {strategy!r}; choose sample-local, team, tensorcore, half2, or auto"
            )
        self.graph = graph
        self.plan = plan
        self.sample_plan = sample_plan
        self.schedule = schedule
        self.weight_offsets = weight_offsets
        self.model_name = _identifier(model_name)
        self.default_batch_tile = default_batch_tile
        self.maximum_batch_tile = maximum_batch_tile

    def _size(self, tensor_id: int) -> int:
        return self.graph.tensors[tensor_id].sample_size

    @staticmethod
    def _scope(lines: list[str]) -> list[str]:
        return ["    {"] + [f"  {line}" for line in lines] + ["    }"]

    def _read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            if tensor_id not in self.weight_offsets:
                raise CompilerError(f"no emitted weight offset for constant tensor {tensor.name!r}")
            return f"weights_({self.weight_offsets[tensor_id]} + {use_index})"
        if tensor_id == self.graph.inputs[0]:
            return f"inputs({use_index})"
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({use_index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no activation storage assigned to tensor {tensor.name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {use_index}]"

    def _write(self, tensor_id: int, index: str) -> str:
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    def _batch_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            if tensor_id not in self.weight_offsets:
                raise CompilerError(f"no emitted weight offset for constant tensor {tensor.name!r}")
            return f"weights({self.weight_offsets[tensor_id]} + {use_index})"
        if tensor_id == self.graph.inputs[0]:
            return f"inputs({use_index},ibatch)"
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({use_index},ibatch)"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no batch-local activation storage assigned to tensor {tensor.name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {use_index}]"

    def _batch_write(self, tensor_id: int, index: str) -> str:
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index},ibatch)"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(
                f"no batch-local activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    def _team_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            if tensor_id not in self.weight_offsets:
                raise CompilerError(f"no emitted weight offset for constant tensor {tensor.name!r}")
            return f"weights({self.weight_offsets[tensor_id]} + {use_index})"
        if tensor_id == self.graph.inputs[0]:
            return f"inputs({use_index},ibatch)"
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({use_index},ibatch)"
        if tensor_id not in self.plan.slots:
            raise CompilerError(f"no hierarchical scratch storage assigned to tensor {tensor.name!r}")
        return f"workspace[({self.plan.slots[tensor_id].offset} + {use_index}) * batch_tile + local_batch]"

    def _team_write(self, tensor_id: int, index: str) -> str:
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index},ibatch)"
        if tensor_id not in self.plan.slots:
            raise CompilerError(
                f"no hierarchical scratch storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[({self.plan.slots[tensor_id].offset} + {index}) * batch_tile + local_batch]"

    def _half_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            if tensor_id not in self.weight_offsets:
                raise CompilerError(f"no emitted half weight offset for constant tensor {tensor.name!r}")
            return f"ponni::TwoHalf::splat(half_weights({self.weight_offsets[tensor_id]} + {use_index}))"
        if tensor_id == self.graph.inputs[0]:
            return f"inputs({use_index})"
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({use_index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no half2 activation storage assigned to tensor {tensor.name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {use_index}]"

    def _half_write(self, tensor_id: int, index: str) -> str:
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(
                f"no half2 activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    @staticmethod
    def _activation(name: str, expression: str, attributes: dict[str, object] | None = None) -> str:
        attributes = attributes or {}
        if name == "LeakyRelu":
            return f"apply_leaky_relu({expression}, static_cast<Scalar>({float(attributes.get('alpha', 0.01))!r}))"
        if name == "Elu":
            return f"apply_elu({expression}, static_cast<Scalar>({float(attributes.get('alpha', 1.0))!r}))"
        if name == "Gelu":
            approximate = str(attributes.get("approximate", "none")) == "tanh"
            return f"apply_gelu({expression}, {str(approximate).lower()})"
        if name == "HardSigmoid":
            alpha = float(attributes.get("alpha", 0.2))
            beta = float(attributes.get("beta", 0.5))
            return f"apply_hard_sigmoid({expression}, static_cast<Scalar>({alpha!r}), static_cast<Scalar>({beta!r}))"
        function = {
            "HardSwish": "apply_hard_swish", "Mish": "apply_mish", "Relu": "apply_relu",
            "Sigmoid": "apply_sigmoid", "Silu": "apply_silu", "Softplus": "apply_softplus", "Tanh": "apply_tanh",
        }.get(name)
        if function is None:
            raise CompilerError(f"C++ emitter has no activation implementation for {name}")
        return f"{function}({expression})"

    @staticmethod
    def _unary(name: str, expression: str, attributes: dict[str, object] | None = None) -> str:
        if name in {"Elu", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Mish", "Relu", "Sigmoid", "Silu",
                    "Softplus", "Tanh"}:
            return CppEmitter._activation(name, expression, attributes)
        function = {"Abs": "Kokkos::abs", "Exp": "Kokkos::exp", "Log": "Kokkos::log", "Sqrt": "Kokkos::sqrt"}.get(name)
        if name == "Neg":
            return f"(-{expression})"
        if function is None:
            raise CompilerError(f"C++ emitter has no unary implementation for {name}")
        return f"{function}({expression})"

    @staticmethod
    def _cuda_activation(name: str, expression: str, attributes: dict[str, object] | None = None) -> str:
        attributes = attributes or {}
        if name == "Relu":
            return f"({expression} > 0.0f ? {expression} : 0.0f)"
        if name == "Sigmoid":
            return f"(1.0f / (1.0f + expf(-{expression})))"
        if name == "Tanh":
            return f"tanhf({expression})"
        if name == "LeakyRelu":
            alpha = float(attributes.get("alpha", 0.01))
            return f"({expression} >= 0.0f ? {expression} : {alpha!r}f * {expression})"
        if name == "Elu":
            alpha = float(attributes.get("alpha", 1.0))
            return f"({expression} >= 0.0f ? {expression} : {alpha!r}f * (expf({expression}) - 1.0f))"
        if name == "Gelu":
            if str(attributes.get("approximate", "none")) == "tanh":
                return (
                    f"(0.5f * {expression} * (1.0f + tanhf(0.7978845608028654f * "
                    f"({expression} + 0.044715f * {expression} * {expression} * {expression}))))"
                )
            return f"(0.5f * {expression} * (1.0f + erff({expression} * 0.7071067811865475f)))"
        if name == "Silu":
            return f"({expression} / (1.0f + expf(-{expression})))"
        if name == "Softplus":
            return f"(fmaxf({expression}, 0.0f) + log1pf(expf(-fabsf({expression}))))"
        if name == "HardSigmoid":
            alpha = float(attributes.get("alpha", 0.2))
            beta = float(attributes.get("beta", 0.5))
            return f"fminf(1.0f, fmaxf(0.0f, {alpha!r}f * {expression} + {beta!r}f))"
        if name == "HardSwish":
            return f"({expression} * fminf(1.0f, fmaxf(0.0f, {expression} / 6.0f + 0.5f)))"
        if name == "Mish":
            return f"({expression} * tanhf(fmaxf({expression}, 0.0f) + log1pf(expf(-fabsf({expression})))))"
        raise CompilerError(f"CUDA emitter has no activation implementation for {name}")

    @staticmethod
    def _binary(op: str, left: str, right: str, half: bool = False) -> str:
        symbol = {"Add": "+", "Div": "/", "Mul": "*", "Sub": "-"}.get(op)
        if symbol is not None:
            return f"({left} {symbol} {right})"
        if half:
            function = {"Max": "maximum", "Min": "minimum", "Pow": "pow"}.get(op)
            if function is not None:
                return f"ponni::TwoHalf::{function}({left}, {right})"
        if op == "Max":
            return f"({left} > {right} ? {left} : {right})"
        if op == "Min":
            return f"({left} < {right} ? {left} : {right})"
        if op == "Pow":
            return f"Kokkos::pow({left}, {right})"
        raise CompilerError(f"C++ emitter has no binary implementation for {op}")

    def _validate_binary(self, node: Node) -> None:
        output_size = self._size(node.outputs[0])
        for tensor_id in node.inputs:
            size = self._size(tensor_id)
            if size not in (1, output_size):
                raise CompilerError(
                    f"unsupported {node.op} broadcasting at {node.source_name or 'canonical node'}: "
                    f"input size {size}, output size {output_size}; only scalar and exact-shape broadcasts are supported"
                )

    @staticmethod
    def _half_unary(name: str, expression: str, attributes: dict[str, object] | None = None) -> str:
        attributes = attributes or {}
        if name == "Neg":
            return f"(ponni::TwoHalf::zero() - {expression})"
        if name == "LeakyRelu":
            return f"ponni::TwoHalf::leaky_relu({expression}, {float(attributes.get('alpha', 0.01))!r}f)"
        if name == "Elu":
            return f"ponni::TwoHalf::elu({expression}, {float(attributes.get('alpha', 1.0))!r}f)"
        if name == "Gelu":
            approximate = str(attributes.get("approximate", "none")) == "tanh"
            return f"ponni::TwoHalf::gelu({expression}, {str(approximate).lower()})"
        if name == "HardSigmoid":
            alpha = float(attributes.get("alpha", 0.2))
            beta = float(attributes.get("beta", 0.5))
            return f"ponni::TwoHalf::hard_sigmoid({expression}, {alpha!r}f, {beta!r}f)"
        function = {
            "Abs": "abs", "Exp": "exp", "HardSwish": "hard_swish", "Log": "log", "Mish": "mish",
            "Relu": "relu", "Sigmoid": "sigmoid", "Silu": "silu", "Softplus": "softplus", "Sqrt": "sqrt",
            "Tanh": "tanh",
        }.get(name)
        if function is None:
            raise CompilerError(f"half2 C++ emitter has no unary implementation for {name}")
        return f"ponni::TwoHalf::{function}({expression})"

    def _emit_node(self, node: Node, batch: bool = False) -> list[str]:
        output_id = node.outputs[0]
        output_size = self._size(output_id)
        read = self._batch_read if batch else self._read
        write = self._batch_write if batch else self._write
        lines: list[str] = []
        if node.op in {"Dense", "DenseBiasActivation"}:
            input_id = node.inputs[0]
            input_size = self._size(input_id)
            weight_id = int(node.attributes["weight"])
            if self._size(weight_id) != output_size * input_size:
                raise CompilerError(f"canonical dense weight size is inconsistent at node {node.id}")
            bias_id = node.attributes.get("bias")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            if bias_id is None:
                lines.append("      Scalar sum = static_cast<Scalar>(0);")
            else:
                lines.append(f"      Scalar sum = {read(int(bias_id), 'i')};")
            lines.append(f"      for (int j = 0; j < {input_size}; j++) {{")
            weight = read(weight_id, f"i * {input_size} + j")
            lines.append(f"        sum += {weight} * {read(input_id, 'j')};")
            lines.append("      }")
            value = "sum"
            if node.op == "DenseBiasActivation":
                value = self._activation(str(node.attributes["activation"]), value,
                                         node.attributes.get("activation_attributes", {}))
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
            self._validate_binary(node)
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._binary(node.op, read(node.inputs[0], "i"), read(node.inputs[1], "i"))
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"Abs", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                        "Neg", "Relu", "Sigmoid", "Silu", "Softplus", "Sqrt", "Tanh"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._unary(node.op, read(node.inputs[0], "i"), node.attributes)
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "Clip":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      Scalar value = {read(node.inputs[0], 'i')};")
            if len(node.inputs) > 1:
                minimum = read(node.inputs[1], "0")
                lines.append(f"      if (value < {minimum}) value = {minimum};")
            elif "min" in node.attributes:
                minimum = f"static_cast<Scalar>({float(node.attributes['min'])!r})"
                lines.append(f"      if (value < {minimum}) value = {minimum};")
            if len(node.inputs) > 2:
                maximum = read(node.inputs[2], "0")
                lines.append(f"      if (value > {maximum}) value = {maximum};")
            elif "max" in node.attributes:
                maximum = f"static_cast<Scalar>({float(node.attributes['max'])!r})"
                lines.append(f"      if (value > {maximum}) value = {maximum};")
            lines.append(f"      {write(output_id, 'i')} = value;")
            lines.append("    }")
            return lines
        if node.op in {"Softmax", "LogSoftmax"}:
            input_id = node.inputs[0]
            lines.append(f"    Scalar maximum = {read(input_id, '0')};")
            lines.append(f"    for (int i = 1; i < {output_size}; i++) {{")
            lines.append(f"      Scalar const value = {read(input_id, 'i')};")
            lines.append("      maximum = value > maximum ? value : maximum;")
            lines.append("    }")
            lines.append("    Scalar exponential_sum = static_cast<Scalar>(0);")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      Scalar const shifted = {read(input_id, 'i')} - maximum;")
            if node.op == "Softmax":
                lines.append("      Scalar const exponential = Kokkos::exp(shifted);")
                lines.append(f"      {write(output_id, 'i')} = exponential;")
            else:
                lines.append(f"      {write(output_id, 'i')} = shifted;")
                lines.append("      Scalar const exponential = Kokkos::exp(shifted);")
            lines.append("      exponential_sum += exponential;")
            lines.append("    }")
            lines.append("    Scalar const normalization = Kokkos::log(exponential_sum);")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            if node.op == "Softmax":
                lines.append(f"      {write(output_id, 'i')} /= exponential_sum;")
            else:
                lines.append(f"      {write(output_id, 'i')} -= normalization;")
            lines.append("    }")
            return lines
        if node.op == "LayerNormalization":
            input_id = node.inputs[0]
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            lines.extend([
                "    Scalar mean = static_cast<Scalar>(0);",
                "    Scalar second_moment = static_cast<Scalar>(0);",
                f"    for (int i = 0; i < {output_size}; i++) {{",
                f"      Scalar const value = {read(input_id, 'i')};",
                "      Scalar const delta = value - mean;",
                "      mean += delta / static_cast<Scalar>(i + 1);",
                "      second_moment += delta * (value - mean);",
                "    }",
                f"    Scalar const variance = second_moment / static_cast<Scalar>({output_size});",
                f"    Scalar const inverse_stddev = static_cast<Scalar>(1) / Kokkos::sqrt(variance + "
                f"static_cast<Scalar>({epsilon!r}));",
                f"    for (int i = 0; i < {output_size}; i++) {{",
            ])
            value = f"({read(input_id, 'i')} - mean) * inverse_stddev * {read(node.inputs[1], 'i')}"
            if len(node.inputs) == 3:
                value = f"({value} + {read(node.inputs[2], 'i')})"
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.extend(["    }",])
            return lines
        if node.op == "BatchNormalization":
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = (
                f"({read(node.inputs[0], 'i')} - {read(node.inputs[3], 'i')}) / "
                f"Kokkos::sqrt({read(node.inputs[4], 'i')} + static_cast<Scalar>({epsilon!r})) * "
                f"{read(node.inputs[1], 'i')} + {read(node.inputs[2], 'i')}"
            )
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"ReduceMean", "ReduceSum"}:
            input_size = self._size(node.inputs[0])
            lines.append("    Scalar reduction = static_cast<Scalar>(0);")
            lines.append(f"    for (int i = 0; i < {input_size}; i++) reduction += {read(node.inputs[0], 'i')};")
            if node.op == "ReduceMean":
                lines.append(f"    reduction /= static_cast<Scalar>({input_size});")
            lines.append(f"    {write(output_id, '0')} = reduction;")
            return lines
        if node.op == "Concat":
            offset = 0
            for input_id in node.inputs:
                input_size = self._size(input_id)
                lines.append(f"    for (int i = 0; i < {input_size}; i++) {{")
                lines.append(f"      {write(output_id, f'{offset} + i')} = {read(input_id, 'i')};")
                lines.append("    }")
                offset += input_size
            if offset != output_size:
                raise CompilerError(f"Concat node {node.id} has inconsistent flattened sample sizes")
            return lines
        if node.op == "ResidualAddActivation":
            self._validate_binary(node)
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            added = self._binary("Add", read(node.inputs[0], "i"), read(node.inputs[1], "i"))
            value = self._activation(str(node.attributes["activation"]), added,
                                     node.attributes.get("activation_attributes", {}))
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "ElementwiseChain":
            for tensor_id in node.inputs:
                size = self._size(tensor_id)
                if size not in (1, output_size):
                    raise CompilerError(f"unsupported elementwise-chain broadcasting from size {size} to {output_size}")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            expression = ""
            for step_index, step in enumerate(node.attributes["steps"]):
                operands = ["value" if value == "prev" else read(int(value), "i")
                            for value in step["inputs"]]
                expression = self._binary(str(step["op"]), operands[0], operands[1])
                declaration = "Scalar value =" if step_index == 0 else "value ="
                lines.append(f"      {declaration} {expression};")
            lines.append(f"      {write(output_id, 'i')} = value;")
            lines.append("    }")
            return lines
        raise CompilerError(f"C++ emitter has no implementation for canonical operation {node.op}")

    def _emit_team_node(self, node: Node) -> list[str]:
        output_id = node.outputs[0]
        output_size = self._size(output_id)
        lines: list[str] = []
        if node.op in {"LayerNormalization", "LogSoftmax", "ReduceMean", "ReduceSum", "Softmax"}:
            lines.append(
                "          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, active_batch), "
                "[&](int local_batch) {"
            )
            lines.append("            int const ibatch = batch_begin + local_batch;")
            lines.append("            (void) ibatch;")
            if node.op in {"Softmax", "LogSoftmax"}:
                input_id = node.inputs[0]
                lines.append(f"            Scalar maximum = {self._team_read(input_id, '0')};")
                lines.append(f"            for (int i = 1; i < {output_size}; i++) {{")
                lines.append(f"              Scalar const value = {self._team_read(input_id, 'i')};")
                lines.append("              maximum = value > maximum ? value : maximum;")
                lines.append("            }")
                lines.append("            Scalar exponential_sum = static_cast<Scalar>(0);")
                lines.append(f"            for (int i = 0; i < {output_size}; i++) {{")
                lines.append(f"              Scalar const shifted = {self._team_read(input_id, 'i')} - maximum;")
                if node.op == "Softmax":
                    lines.append("              Scalar const exponential = Kokkos::exp(shifted);")
                    lines.append(f"              {self._team_write(output_id, 'i')} = exponential;")
                else:
                    lines.append(f"              {self._team_write(output_id, 'i')} = shifted;")
                    lines.append("              Scalar const exponential = Kokkos::exp(shifted);")
                lines.append("              exponential_sum += exponential;")
                lines.append("            }")
                lines.append("            Scalar const normalization = Kokkos::log(exponential_sum);")
                lines.append(f"            for (int i = 0; i < {output_size}; i++) {{")
                operator = "/= exponential_sum" if node.op == "Softmax" else "-= normalization"
                lines.append(f"              {self._team_write(output_id, 'i')} {operator};")
                lines.append("            }")
            elif node.op == "LayerNormalization":
                input_id = node.inputs[0]
                epsilon = float(node.attributes.get("epsilon", 1e-5))
                lines.extend([
                    "            Scalar mean = static_cast<Scalar>(0);",
                    "            Scalar second_moment = static_cast<Scalar>(0);",
                    f"            for (int i = 0; i < {output_size}; i++) {{",
                    f"              Scalar const value = {self._team_read(input_id, 'i')};",
                    "              Scalar const delta = value - mean;",
                    "              mean += delta / static_cast<Scalar>(i + 1);",
                    "              second_moment += delta * (value - mean);",
                    "            }",
                    f"            Scalar const variance = second_moment / static_cast<Scalar>({output_size});",
                    "            Scalar const inverse_stddev = static_cast<Scalar>(1) / "
                    f"Kokkos::sqrt(variance + static_cast<Scalar>({epsilon!r}));",
                    f"            for (int i = 0; i < {output_size}; i++) {{",
                ])
                value = (
                    f"({self._team_read(input_id, 'i')} - mean) * inverse_stddev * "
                    f"{self._team_read(node.inputs[1], 'i')}"
                )
                if len(node.inputs) == 3:
                    value = f"({value} + {self._team_read(node.inputs[2], 'i')})"
                lines.append(f"              {self._team_write(output_id, 'i')} = {value};")
                lines.append("            }")
            else:
                input_size = self._size(node.inputs[0])
                lines.append("            Scalar reduction = static_cast<Scalar>(0);")
                lines.append(
                    f"            for (int i = 0; i < {input_size}; i++) reduction += "
                    f"{self._team_read(node.inputs[0], 'i')};"
                )
                if node.op == "ReduceMean":
                    lines.append(f"            reduction /= static_cast<Scalar>({input_size});")
                lines.append(f"            {self._team_write(output_id, '0')} = reduction;")
            lines.extend(["          });", "          team.team_barrier();"])
            return lines
        lines.append(
            f"          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, {output_size} * active_batch), "
            "[&](int linear) {"
        )
        lines.append("            int const local_batch = linear % active_batch;")
        lines.append("            int const i = linear / active_batch;")
        lines.append("            int const ibatch = batch_begin + local_batch;")
        lines.append("            (void) ibatch;")
        if node.op in {"Dense", "DenseBiasActivation"}:
            input_id = node.inputs[0]
            input_size = self._size(input_id)
            weight_id = int(node.attributes["weight"])
            if self._size(weight_id) != output_size * input_size:
                raise CompilerError(f"canonical dense weight size is inconsistent at node {node.id}")
            bias_id = node.attributes.get("bias")
            if bias_id is None:
                lines.append("            Scalar sum = static_cast<Scalar>(0);")
            else:
                lines.append(f"            Scalar sum = {self._team_read(int(bias_id), 'i')};")
            lines.append(f"            for (int j = 0; j < {input_size}; j++) {{")
            weight = self._team_read(weight_id, f"i * {input_size} + j")
            lines.append(f"              sum += {weight} * {self._team_read(input_id, 'j')};")
            lines.append("            }")
            value = "sum"
            if node.op == "DenseBiasActivation":
                value = self._activation(str(node.attributes["activation"]), value,
                                         node.attributes.get("activation_attributes", {}))
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
            self._validate_binary(node)
            value = self._binary(
                node.op, self._team_read(node.inputs[0], "i"), self._team_read(node.inputs[1], "i")
            )
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op in {"Abs", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                            "Neg", "Relu", "Sigmoid", "Silu", "Softplus", "Sqrt", "Tanh"}:
            value = self._unary(node.op, self._team_read(node.inputs[0], "i"), node.attributes)
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "Clip":
            lines.append(f"            Scalar value = {self._team_read(node.inputs[0], 'i')};")
            if len(node.inputs) > 1:
                minimum = self._team_read(node.inputs[1], "0")
                lines.append(f"            if (value < {minimum}) value = {minimum};")
            elif "min" in node.attributes:
                minimum = f"static_cast<Scalar>({float(node.attributes['min'])!r})"
                lines.append(f"            if (value < {minimum}) value = {minimum};")
            if len(node.inputs) > 2:
                maximum = self._team_read(node.inputs[2], "0")
                lines.append(f"            if (value > {maximum}) value = {maximum};")
            elif "max" in node.attributes:
                maximum = f"static_cast<Scalar>({float(node.attributes['max'])!r})"
                lines.append(f"            if (value > {maximum}) value = {maximum};")
            lines.append(f"            {self._team_write(output_id, 'i')} = value;")
        elif node.op == "BatchNormalization":
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            value = (
                f"({self._team_read(node.inputs[0], 'i')} - {self._team_read(node.inputs[3], 'i')}) / "
                f"Kokkos::sqrt({self._team_read(node.inputs[4], 'i')} + static_cast<Scalar>({epsilon!r})) * "
                f"{self._team_read(node.inputs[1], 'i')} + {self._team_read(node.inputs[2], 'i')}"
            )
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "Concat":
            offset = 0
            for input_index, input_id in enumerate(node.inputs):
                input_size = self._size(input_id)
                condition = f"i < {offset + input_size}" if input_index == 0 else f"i < {offset + input_size}"
                keyword = "if" if input_index == 0 else "else if"
                lines.append(f"            {keyword} ({condition}) {{")
                lines.append(
                    f"              {self._team_write(output_id, 'i')} = "
                    f"{self._team_read(input_id, f'i - {offset}')};"
                )
                lines.append("            }")
                offset += input_size
            if offset != output_size:
                raise CompilerError(f"Concat node {node.id} has inconsistent flattened sample sizes")
        elif node.op == "ResidualAddActivation":
            self._validate_binary(node)
            added = self._binary(
                "Add", self._team_read(node.inputs[0], "i"), self._team_read(node.inputs[1], "i")
            )
            value = self._activation(str(node.attributes["activation"]), added,
                                     node.attributes.get("activation_attributes", {}))
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "ElementwiseChain":
            for tensor_id in node.inputs:
                size = self._size(tensor_id)
                if size not in (1, output_size):
                    raise CompilerError(f"unsupported elementwise-chain broadcasting from size {size} to {output_size}")
            expression = ""
            for step_index, step in enumerate(node.attributes["steps"]):
                operands = [
                    "value" if value == "prev" else self._team_read(int(value), "i")
                    for value in step["inputs"]
                ]
                expression = self._binary(str(step["op"]), operands[0], operands[1])
                declaration = "Scalar value =" if step_index == 0 else "value ="
                lines.append(f"            {declaration} {expression};")
            lines.append(f"            {self._team_write(output_id, 'i')} = value;")
        else:
            raise CompilerError(f"hierarchical C++ emitter has no implementation for canonical operation {node.op}")
        lines.append("          });")
        lines.append("          team.team_barrier();")
        return lines

    def _emit_half_node(self, node: Node, accumulator_count: int = 0) -> list[str]:
        output_id = node.outputs[0]
        output_size = self._size(output_id)
        lines: list[str] = []
        if node.op in {"Dense", "DenseBiasActivation"}:
            input_id = node.inputs[0]
            input_size = self._size(input_id)
            weight_id = int(node.attributes["weight"])
            bias_id = node.attributes.get("bias")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            initial = "ponni::TwoHalf::zero()" if bias_id is None else self._half_read(int(bias_id), "i")
            weight = self._half_read(weight_id, f"i * {input_size} + j")
            if accumulator_count == 0:
                lines.append(f"      ponni::TwoHalf sum = {initial};")
                lines.append(f"      for (int j = 0; j < {input_size}; j++) {{")
                lines.append(f"        sum = ponni::TwoHalf::fma({weight}, {self._half_read(input_id, 'j')}, sum);")
                lines.append("      }")
            else:
                for iaccumulator in range(accumulator_count):
                    lines.append(
                        f"      ponni::TwoHalf sum_{iaccumulator} = ponni::TwoHalf::zero();"
                    )
                lines.append("      int j = 0;")
                lines.append(f"      for (; j + {accumulator_count} <= {input_size}; j += {accumulator_count}) {{")
                for iaccumulator in range(accumulator_count):
                    offset = f"j + {iaccumulator}"
                    weight = self._half_read(weight_id, f"i * {input_size} + {offset}")
                    value = self._half_read(input_id, offset)
                    lines.append(
                        f"        sum_{iaccumulator} = ponni::TwoHalf::fma({weight}, {value}, "
                        f"sum_{iaccumulator});"
                    )
                lines.append("      }")
                for iaccumulator in range(accumulator_count):
                    lines.append(f"      if (j + {iaccumulator} < {input_size}) {{")
                    weight = self._half_read(weight_id, f"i * {input_size} + j + {iaccumulator}")
                    value = self._half_read(input_id, f"j + {iaccumulator}")
                    lines.append(
                        f"        sum_{iaccumulator} = ponni::TwoHalf::fma({weight}, {value}, "
                        f"sum_{iaccumulator});"
                    )
                    lines.append("      }")
                lines.append(f"      ponni::TwoHalf const bias = {initial};")
                low_terms = " + ".join(f"sum_{index}.low()" for index in range(accumulator_count))
                high_terms = " + ".join(f"sum_{index}.high()" for index in range(accumulator_count))
                lines.append(
                    f"      ponni::TwoHalf sum = ponni::TwoHalf::from_floats(bias.low() + {low_terms}, "
                    f"bias.high() + {high_terms});"
                )
            if node.op == "DenseBiasActivation":
                value = self._half_unary(str(node.attributes["activation"]), "sum",
                                         node.attributes.get("activation_attributes", {}))
            else:
                value = "sum"
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}:
            self._validate_binary(node)
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._binary(node.op, self._half_read(node.inputs[0], "i"),
                                 self._half_read(node.inputs[1], "i"), half=True)
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"Abs", "Elu", "Exp", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                        "Neg", "Relu", "Sigmoid", "Silu", "Softplus", "Sqrt", "Tanh"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._half_unary(node.op, self._half_read(node.inputs[0], "i"), node.attributes)
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "Clip":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      ponni::TwoHalf value = {self._half_read(node.inputs[0], 'i')};")
            if len(node.inputs) > 1:
                lines.append(
                    f"      value = ponni::TwoHalf::maximum(value, {self._half_read(node.inputs[1], '0')});"
                )
            elif "min" in node.attributes:
                minimum = float(node.attributes["min"])
                lines.append(
                    f"      value = ponni::TwoHalf::maximum(value, ponni::TwoHalf::from_floats({minimum!r}f, "
                    f"{minimum!r}f));"
                )
            if len(node.inputs) > 2:
                lines.append(
                    f"      value = ponni::TwoHalf::minimum(value, {self._half_read(node.inputs[2], '0')});"
                )
            elif "max" in node.attributes:
                maximum = float(node.attributes["max"])
                lines.append(
                    f"      value = ponni::TwoHalf::minimum(value, ponni::TwoHalf::from_floats({maximum!r}f, "
                    f"{maximum!r}f));"
                )
            lines.append(f"      {self._half_write(output_id, 'i')} = value;")
            lines.append("    }")
            return lines
        if node.op in {"Softmax", "LogSoftmax"}:
            input_id = node.inputs[0]
            lines.append(f"    ponni::TwoHalf maximum = {self._half_read(input_id, '0')};")
            lines.append(f"    for (int i = 1; i < {output_size}; i++) {{")
            lines.append(f"      maximum = ponni::TwoHalf::maximum(maximum, {self._half_read(input_id, 'i')});")
            lines.append("    }")
            lines.append("    ponni::TwoHalf exponential_sum = ponni::TwoHalf::zero();")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      ponni::TwoHalf const shifted = {self._half_read(input_id, 'i')} - maximum;")
            if node.op == "Softmax":
                lines.append("      ponni::TwoHalf const exponential = ponni::TwoHalf::exp(shifted);")
                lines.append(f"      {self._half_write(output_id, 'i')} = exponential;")
            else:
                lines.append(f"      {self._half_write(output_id, 'i')} = shifted;")
                lines.append("      ponni::TwoHalf const exponential = ponni::TwoHalf::exp(shifted);")
            lines.append("      exponential_sum = exponential_sum + exponential;")
            lines.append("    }")
            lines.append("    ponni::TwoHalf const normalization = ponni::TwoHalf::log(exponential_sum);")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            operator = "/ exponential_sum" if node.op == "Softmax" else "- normalization"
            lines.append(f"      {self._half_write(output_id, 'i')} = {self._half_write(output_id, 'i')} {operator};")
            lines.append("    }")
            return lines
        if node.op == "LayerNormalization":
            input_id = node.inputs[0]
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            lines.extend([
                "    float mean_low = 0.0f;",
                "    float mean_high = 0.0f;",
                "    float second_moment_low = 0.0f;",
                "    float second_moment_high = 0.0f;",
                f"    for (int i = 0; i < {output_size}; i++) {{",
                f"      ponni::TwoHalf const value = {self._half_read(input_id, 'i')};",
                "      float const delta_low = value.low() - mean_low;",
                "      float const delta_high = value.high() - mean_high;",
                "      mean_low += delta_low / static_cast<float>(i + 1);",
                "      mean_high += delta_high / static_cast<float>(i + 1);",
                "      second_moment_low += delta_low * (value.low() - mean_low);",
                "      second_moment_high += delta_high * (value.high() - mean_high);",
                "    }",
                "    ponni::TwoHalf const mean = ponni::TwoHalf::from_floats(mean_low, mean_high);",
                "    ponni::TwoHalf const inverse_stddev = ponni::TwoHalf::from_floats("
                f"1.0f / Kokkos::sqrt(second_moment_low / {output_size}.0f + {epsilon!r}f), "
                f"1.0f / Kokkos::sqrt(second_moment_high / {output_size}.0f + {epsilon!r}f));",
                f"    for (int i = 0; i < {output_size}; i++) {{",
            ])
            value = (
                f"({self._half_read(input_id, 'i')} - mean) * inverse_stddev * "
                f"{self._half_read(node.inputs[1], 'i')}"
            )
            if len(node.inputs) == 3:
                value = f"({value} + {self._half_read(node.inputs[2], 'i')})"
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "BatchNormalization":
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            epsilon_value = f"ponni::TwoHalf::from_floats({epsilon!r}f, {epsilon!r}f)"
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = (
                f"({self._half_read(node.inputs[0], 'i')} - {self._half_read(node.inputs[3], 'i')}) / "
                f"ponni::TwoHalf::sqrt({self._half_read(node.inputs[4], 'i')} + {epsilon_value}) * "
                f"{self._half_read(node.inputs[1], 'i')} + {self._half_read(node.inputs[2], 'i')}"
            )
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"ReduceMean", "ReduceSum"}:
            input_size = self._size(node.inputs[0])
            lines.append("    ponni::TwoHalf reduction = ponni::TwoHalf::zero();")
            lines.append(
                f"    for (int i = 0; i < {input_size}; i++) reduction = reduction + "
                f"{self._half_read(node.inputs[0], 'i')};"
            )
            if node.op == "ReduceMean":
                lines.append(
                    f"    reduction = reduction / ponni::TwoHalf::from_floats({input_size}.0f, {input_size}.0f);"
                )
            lines.append(f"    {self._half_write(output_id, '0')} = reduction;")
            return lines
        if node.op == "Concat":
            offset = 0
            for input_id in node.inputs:
                input_size = self._size(input_id)
                lines.append(f"    for (int i = 0; i < {input_size}; i++) {{")
                lines.append(
                    f"      {self._half_write(output_id, f'{offset} + i')} = {self._half_read(input_id, 'i')};"
                )
                lines.append("    }")
                offset += input_size
            if offset != output_size:
                raise CompilerError(f"Concat node {node.id} has inconsistent flattened sample sizes")
            return lines
        if node.op == "ResidualAddActivation":
            self._validate_binary(node)
            added = self._binary("Add", self._half_read(node.inputs[0], "i"),
                                 self._half_read(node.inputs[1], "i"), half=True)
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._half_unary(str(node.attributes["activation"]), added,
                                     node.attributes.get("activation_attributes", {}))
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "ElementwiseChain":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            expression = ""
            for step_index, step in enumerate(node.attributes["steps"]):
                operands = ["value" if value == "prev" else self._half_read(int(value), "i")
                            for value in step["inputs"]]
                expression = self._binary(str(step["op"]), operands[0], operands[1], half=True)
                declaration = "ponni::TwoHalf value =" if step_index == 0 else "value ="
                lines.append(f"      {declaration} {expression};")
            lines.append(f"      {self._half_write(output_id, 'i')} = value;")
            lines.append("    }")
            return lines
        raise CompilerError(f"half2 C++ emitter has no implementation for canonical operation {node.op}")

    def _emit_streaming_dense_pair(self, producer: Node, consumer: Node, batch: bool) -> list[str]:
        read = self._batch_read if batch else self._read
        write = self._batch_write if batch else self._write
        input_id = producer.inputs[0]
        hidden_id = producer.outputs[0]
        output_id = consumer.outputs[0]
        input_size = self._size(input_id)
        hidden_size = self._size(hidden_id)
        output_size = self._size(output_id)
        producer_weight = int(producer.attributes["weight"])
        consumer_weight = int(consumer.attributes["weight"])
        producer_bias = producer.attributes.get("bias")
        consumer_bias = consumer.attributes.get("bias")
        lines: list[str] = []
        cache_batch_inputs = batch and input_size <= 16
        if cache_batch_inputs:
            for iinput in range(input_size):
                lines.append(f"    Scalar const input_{iinput} = {read(input_id, str(iinput))};")
        for ioutput in range(output_size):
            initial = ("static_cast<Scalar>(0)" if consumer_bias is None
                       else read(int(consumer_bias), str(ioutput)))
            lines.append(f"    Scalar output_accumulator_{ioutput} = {initial};")
        lines.append(f"    for (int ihidden = 0; ihidden < {hidden_size}; ihidden++) {{")
        hidden_initial = ("static_cast<Scalar>(0)" if producer_bias is None
                          else read(int(producer_bias), "ihidden"))
        lines.append(f"      Scalar hidden = {hidden_initial};")
        if cache_batch_inputs:
            for iinput in range(input_size):
                weight_value = read(producer_weight, f"ihidden * {input_size} + {iinput}")
                lines.append(f"      hidden += {weight_value} * input_{iinput};")
        else:
            lines.append(f"      for (int iinput = 0; iinput < {input_size}; iinput++) {{")
            weight_value = read(producer_weight, f"ihidden * {input_size} + iinput")
            lines.append(f"        hidden += {weight_value} * {read(input_id, 'iinput')};")
            lines.append("      }")
        lines.append(
            f"      hidden = {self._activation(str(producer.attributes['activation']), 'hidden', producer.attributes.get('activation_attributes', {}))};"
        )
        for ioutput in range(output_size):
            weight_value = read(consumer_weight, f"{ioutput} * {hidden_size} + ihidden")
            lines.append(f"      output_accumulator_{ioutput} += {weight_value} * hidden;")
        lines.append("    }")
        for ioutput in range(output_size):
            value = f"output_accumulator_{ioutput}"
            if consumer.op == "DenseBiasActivation":
                value = self._activation(str(consumer.attributes["activation"]), value,
                                         consumer.attributes.get("activation_attributes", {}))
            lines.append(f"    {write(output_id, str(ioutput))} = {value};")
        return lines

    def _emit_half_streaming_dense_pair(self, producer: Node, consumer: Node,
                                        producer_accumulator_count: int = 0,
                                        consumer_accumulator_count: int = 0) -> list[str]:
        input_id = producer.inputs[0]
        hidden_id = producer.outputs[0]
        output_id = consumer.outputs[0]
        input_size = self._size(input_id)
        hidden_size = self._size(hidden_id)
        output_size = self._size(output_id)
        producer_weight = int(producer.attributes["weight"])
        consumer_weight = int(consumer.attributes["weight"])
        producer_bias = producer.attributes.get("bias")
        consumer_bias = consumer.attributes.get("bias")
        lines: list[str] = []
        for ioutput in range(output_size):
            initial = ("ponni::TwoHalf::zero()" if consumer_bias is None
                       else self._half_read(int(consumer_bias), str(ioutput)))
            if consumer_accumulator_count == 0:
                lines.append(f"    ponni::TwoHalf output_accumulator_{ioutput} = {initial};")
            else:
                lines.append(f"    ponni::TwoHalf const output_bias_{ioutput} = {initial};")
                for iaccumulator in range(consumer_accumulator_count):
                    lines.append(
                        f"    ponni::TwoHalf output_accumulator_{ioutput}_{iaccumulator} = "
                        "ponni::TwoHalf::zero();"
                    )
        lines.append(f"    for (int ihidden = 0; ihidden < {hidden_size}; ihidden++) {{")
        hidden_initial = ("ponni::TwoHalf::zero()" if producer_bias is None
                          else self._half_read(int(producer_bias), "ihidden"))
        if producer_accumulator_count == 0:
            lines.append(f"      ponni::TwoHalf hidden = {hidden_initial};")
            lines.append(f"      for (int iinput = 0; iinput < {input_size}; iinput++) {{")
            producer_value = self._half_read(producer_weight, f"ihidden * {input_size} + iinput")
            lines.append(
                f"        hidden = ponni::TwoHalf::fma({producer_value}, "
                f"{self._half_read(input_id, 'iinput')}, hidden);"
            )
            lines.append("      }")
        else:
            for iaccumulator in range(producer_accumulator_count):
                lines.append(
                    f"      ponni::TwoHalf hidden_{iaccumulator} = ponni::TwoHalf::zero();"
                )
            lines.append("      int iinput = 0;")
            lines.append(
                f"      for (; iinput + {producer_accumulator_count} <= {input_size}; "
                f"iinput += {producer_accumulator_count}) {{"
            )
            for iaccumulator in range(producer_accumulator_count):
                offset = f"iinput + {iaccumulator}"
                producer_value = self._half_read(producer_weight, f"ihidden * {input_size} + {offset}")
                lines.append(
                    f"        hidden_{iaccumulator} = ponni::TwoHalf::fma({producer_value}, "
                    f"{self._half_read(input_id, offset)}, hidden_{iaccumulator});"
                )
            lines.append("      }")
            for iaccumulator in range(producer_accumulator_count):
                lines.append(f"      if (iinput + {iaccumulator} < {input_size}) {{")
                producer_value = self._half_read(
                    producer_weight, f"ihidden * {input_size} + iinput + {iaccumulator}"
                )
                lines.append(
                    f"        hidden_{iaccumulator} = ponni::TwoHalf::fma({producer_value}, "
                    f"{self._half_read(input_id, f'iinput + {iaccumulator}')}, hidden_{iaccumulator});"
                )
                lines.append("      }")
            lines.append(f"      ponni::TwoHalf const hidden_bias = {hidden_initial};")
            low_terms = " + ".join(
                f"hidden_{index}.low()" for index in range(producer_accumulator_count)
            )
            high_terms = " + ".join(
                f"hidden_{index}.high()" for index in range(producer_accumulator_count)
            )
            lines.append(
                f"      ponni::TwoHalf hidden = ponni::TwoHalf::from_floats(hidden_bias.low() + {low_terms}, "
                f"hidden_bias.high() + {high_terms});"
            )
        lines.append(
            f"      hidden = {self._half_unary(str(producer.attributes['activation']), 'hidden', producer.attributes.get('activation_attributes', {}))};"
        )
        for ioutput in range(output_size):
            weight = self._half_read(consumer_weight, f"{ioutput} * {hidden_size} + ihidden")
            if consumer_accumulator_count == 0:
                lines.append(
                    f"      output_accumulator_{ioutput} = ponni::TwoHalf::fma("
                    f"{weight}, hidden, output_accumulator_{ioutput});"
                )
            else:
                for iaccumulator in range(consumer_accumulator_count):
                    keyword = "if" if iaccumulator == 0 else "else if"
                    lines.append(
                        f"      {keyword} ((ihidden & {consumer_accumulator_count - 1}) == {iaccumulator}) {{"
                    )
                    lines.append(
                        f"        output_accumulator_{ioutput}_{iaccumulator} = ponni::TwoHalf::fma("
                        f"{weight}, hidden, output_accumulator_{ioutput}_{iaccumulator});"
                    )
                    lines.append("      }")
        lines.append("    }")
        for ioutput in range(output_size):
            if consumer_accumulator_count == 0:
                value = f"output_accumulator_{ioutput}"
            else:
                low_terms = " + ".join(
                    f"output_accumulator_{ioutput}_{index}.low()"
                    for index in range(consumer_accumulator_count)
                )
                high_terms = " + ".join(
                    f"output_accumulator_{ioutput}_{index}.high()"
                    for index in range(consumer_accumulator_count)
                )
                lines.append(
                    f"    ponni::TwoHalf output_accumulator_{ioutput} = ponni::TwoHalf::from_floats("
                    f"output_bias_{ioutput}.low() + {low_terms}, output_bias_{ioutput}.high() + {high_terms});"
                )
                value = f"output_accumulator_{ioutput}"
            if consumer.op == "DenseBiasActivation":
                value = self._half_unary(str(consumer.attributes["activation"]), value,
                                         consumer.attributes.get("activation_attributes", {}))
            lines.append(f"    {self._half_write(output_id, str(ioutput))} = {value};")
        return lines

    def _emit_tensorcore_dense_pair(self, producer: Node, consumer: Node) -> str:
        input_size = self._size(producer.inputs[0])
        hidden_size = self._size(producer.outputs[0])
        output_size = self._size(consumer.outputs[0])
        producer_weight = int(producer.attributes["weight"])
        consumer_weight = int(consumer.attributes["weight"])
        producer_bias = producer.attributes.get("bias")
        consumer_bias = consumer.attributes.get("bias")
        producer_weight_offset = self.weight_offsets[producer_weight]
        consumer_weight_offset = self.weight_offsets[consumer_weight]
        producer_bias_value = (
            "0.0f" if producer_bias is None
            else f"weights[{self.weight_offsets[int(producer_bias)]} + hidden_neuron]"
        )
        consumer_bias_value = (
            "0.0f" if consumer_bias is None
            else f"weights[{self.weight_offsets[int(consumer_bias)]} + output_neuron]"
        )
        hidden_activation = self._cuda_activation(
            str(producer.attributes["activation"]), "hidden_value",
            producer.attributes.get("activation_attributes", {}),
        )
        output_value = "output_tile[output_neuron * tensorcore_batch_tile + local_batch]"
        if consumer.op == "DenseBiasActivation":
            output_value = self._cuda_activation(
                str(consumer.attributes["activation"]), output_value,
                consumer.attributes.get("activation_attributes", {}),
            )
        kernel_name = f"{self.model_name}_tensorcore_kernel"
        return f"""#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
static __global__ void {kernel_name}(float const * inputs, float * outputs,
                                     float const * weights, int batch_size) {{
  int constexpr tensorcore_batch_tile = 16;
  int constexpr input_tile_elements = 8 * tensorcore_batch_tile;
  int constexpr weight_tile_elements = 16 * 8;
  int constexpr matrix_tile_elements = 16 * tensorcore_batch_tile;
  int constexpr scratch_elements_per_warp =
      input_tile_elements + weight_tile_elements + matrix_tile_elements;
  int const lane = threadIdx.x & 31;
  int const warp_in_block = threadIdx.x >> 5;
  int const warps_per_block = blockDim.x >> 5;
  int const batch_begin = (blockIdx.x * warps_per_block + warp_in_block) * tensorcore_batch_tile;
  if (batch_begin >= batch_size) return;

  extern __shared__ __align__(32) unsigned char dynamic_scratch[];
  float * input_tile = reinterpret_cast<float *>(dynamic_scratch) + warp_in_block * scratch_elements_per_warp;
  float * weight_tile = input_tile + input_tile_elements;
  float * matrix_tile = weight_tile + weight_tile_elements;

  for (int linear = lane; linear < input_tile_elements; linear += 32) {{
    int const input_feature = linear / tensorcore_batch_tile;
    int const local_batch = linear % tensorcore_batch_tile;
    int const ibatch = batch_begin + local_batch;
    float const input_value = input_feature < {input_size} && ibatch < batch_size
                                ? inputs[input_feature * batch_size + ibatch] : 0.0f;
    input_tile[linear] = nvcuda::wmma::__float_to_tf32(input_value);
  }}
  __syncwarp();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> output_fragment;
  nvcuda::wmma::fill_fragment(output_fragment, 0.0f);
  for (int hidden_begin = 0; hidden_begin < {hidden_size}; hidden_begin += 16) {{
    for (int linear = lane; linear < weight_tile_elements; linear += 32) {{
      int const local_hidden = linear / 8;
      int const input_feature = linear % 8;
      int const hidden_neuron = hidden_begin + local_hidden;
      float const weight_value = hidden_neuron < {hidden_size} && input_feature < {input_size}
                                   ? weights[{producer_weight_offset} + hidden_neuron * {input_size} + input_feature]
                                   : 0.0f;
      weight_tile[linear] = nvcuda::wmma::__float_to_tf32(weight_value);
    }}
    __syncwarp();

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
        input_weight_fragment;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
        input_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> hidden_fragment;
    nvcuda::wmma::load_matrix_sync(input_weight_fragment, weight_tile, 8);
    nvcuda::wmma::load_matrix_sync(input_fragment, input_tile, tensorcore_batch_tile);
    nvcuda::wmma::fill_fragment(hidden_fragment, 0.0f);
    nvcuda::wmma::mma_sync(hidden_fragment, input_weight_fragment, input_fragment, hidden_fragment);
    nvcuda::wmma::store_matrix_sync(
        matrix_tile, hidden_fragment, tensorcore_batch_tile, nvcuda::wmma::mem_row_major);
    __syncwarp();

    for (int linear = lane; linear < matrix_tile_elements; linear += 32) {{
      int const local_hidden = linear / tensorcore_batch_tile;
      int const hidden_neuron = hidden_begin + local_hidden;
      float hidden_value = matrix_tile[linear];
      if (hidden_neuron < {hidden_size}) hidden_value += {producer_bias_value};
      hidden_value = hidden_neuron < {hidden_size} ? {hidden_activation} : 0.0f;
      matrix_tile[linear] = nvcuda::wmma::__float_to_tf32(hidden_value);
    }}
    __syncwarp();

    for (int hidden_half = 0; hidden_half < 2; hidden_half++) {{
      for (int linear = lane; linear < weight_tile_elements; linear += 32) {{
        int const output_neuron = linear / 8;
        int const local_hidden = hidden_half * 8 + linear % 8;
        int const hidden_neuron = hidden_begin + local_hidden;
        float const weight_value = output_neuron < {output_size} && hidden_neuron < {hidden_size}
                                     ? weights[{consumer_weight_offset} + output_neuron * {hidden_size} + hidden_neuron]
                                     : 0.0f;
        weight_tile[linear] = nvcuda::wmma::__float_to_tf32(weight_value);
      }}
      __syncwarp();

      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          output_weight_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          hidden_input_fragment;
      nvcuda::wmma::load_matrix_sync(output_weight_fragment, weight_tile, 8);
      nvcuda::wmma::load_matrix_sync(
          hidden_input_fragment,
          matrix_tile + hidden_half * 8 * tensorcore_batch_tile,
          tensorcore_batch_tile);
      nvcuda::wmma::mma_sync(
          output_fragment, output_weight_fragment, hidden_input_fragment, output_fragment);
      __syncwarp();
    }}
  }}

  float * output_tile = matrix_tile;
  nvcuda::wmma::store_matrix_sync(
      output_tile, output_fragment, tensorcore_batch_tile, nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int linear = lane; linear < {output_size} * tensorcore_batch_tile; linear += 32) {{
    int const output_neuron = linear / tensorcore_batch_tile;
    int const local_batch = linear % tensorcore_batch_tile;
    int const ibatch = batch_begin + local_batch;
    if (ibatch < batch_size) {{
      output_tile[linear] += {consumer_bias_value};
      outputs[output_neuron * batch_size + ibatch] = {output_value};
    }}
  }}
}}
#endif"""

    def _emit_tensorcore_dense_triple(self, first: Node, second: Node, consumer: Node) -> str:
        input_size = self._size(first.inputs[0])
        first_size = self._size(first.outputs[0])
        second_size = self._size(second.outputs[0])
        output_size = self._size(consumer.outputs[0])
        first_weight = int(first.attributes["weight"])
        second_weight = int(second.attributes["weight"])
        consumer_weight = int(consumer.attributes["weight"])

        def bias_value(node: Node, index: str) -> str:
            bias = node.attributes.get("bias")
            return "0.0f" if bias is None else f"weights[{self.weight_offsets[int(bias)]} + {index}]"

        first_bias = bias_value(first, "first_neuron")
        second_bias = bias_value(second, "second_neuron")
        consumer_bias = bias_value(consumer, "output_neuron")
        first_activation = self._cuda_activation(
            str(first.attributes["activation"]), "activation_value", first.attributes.get("activation_attributes", {})
        )
        second_activation = self._cuda_activation(
            str(second.attributes["activation"]), "activation_value", second.attributes.get("activation_attributes", {})
        )
        output_value = "matrix_tile[output_neuron * tensorcore_batch_tile + local_batch]"
        if consumer.op == "DenseBiasActivation":
            output_value = self._cuda_activation(
                str(consumer.attributes["activation"]), output_value,
                consumer.attributes.get("activation_attributes", {}),
            )
        kernel_name = f"{self.model_name}_tensorcore_kernel"
        return f"""#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
static __global__ void {kernel_name}(float const * inputs, float * outputs,
                                     float const * weights, int batch_size) {{
  int constexpr tensorcore_batch_tile = 16;
  int constexpr padded_inputs = {((input_size + 7) // 8) * 8};
  int constexpr padded_first = {((first_size + 7) // 8) * 8};
  int constexpr input_tile_elements = padded_inputs * tensorcore_batch_tile;
  int constexpr first_tile_elements = padded_first * tensorcore_batch_tile;
  int constexpr weight_tile_elements = 16 * 8;
  int constexpr matrix_tile_elements = 16 * tensorcore_batch_tile;
  int constexpr scratch_elements_per_warp =
      input_tile_elements + first_tile_elements + weight_tile_elements + matrix_tile_elements;
  int const lane = threadIdx.x & 31;
  int const warp_in_block = threadIdx.x >> 5;
  int const warps_per_block = blockDim.x >> 5;
  int const batch_begin = (blockIdx.x * warps_per_block + warp_in_block) * tensorcore_batch_tile;
  if (batch_begin >= batch_size) return;

  extern __shared__ __align__(32) unsigned char dynamic_scratch[];
  float * input_tile = reinterpret_cast<float *>(dynamic_scratch) + warp_in_block * scratch_elements_per_warp;
  float * first_tile = input_tile + input_tile_elements;
  float * weight_tile = first_tile + first_tile_elements;
  float * matrix_tile = weight_tile + weight_tile_elements;

  for (int linear = lane; linear < input_tile_elements; linear += 32) {{
    int const input_feature = linear / tensorcore_batch_tile;
    int const local_batch = linear % tensorcore_batch_tile;
    int const ibatch = batch_begin + local_batch;
    float const value = input_feature < {input_size} && ibatch < batch_size
                            ? inputs[input_feature * batch_size + ibatch] : 0.0f;
    input_tile[linear] = nvcuda::wmma::__float_to_tf32(value);
  }}
  __syncwarp();

  for (int first_begin = 0; first_begin < {first_size}; first_begin += 16) {{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> first_fragment;
    nvcuda::wmma::fill_fragment(first_fragment, 0.0f);
    for (int input_begin = 0; input_begin < padded_inputs; input_begin += 8) {{
      for (int linear = lane; linear < weight_tile_elements; linear += 32) {{
        int const first_neuron = first_begin + linear / 8;
        int const input_feature = input_begin + linear % 8;
        float const value = first_neuron < {first_size} && input_feature < {input_size}
                                ? weights[{self.weight_offsets[first_weight]} +
                                          first_neuron * {input_size} + input_feature] : 0.0f;
        weight_tile[linear] = nvcuda::wmma::__float_to_tf32(value);
      }}
      __syncwarp();
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          weight_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          input_fragment;
      nvcuda::wmma::load_matrix_sync(weight_fragment, weight_tile, 8);
      nvcuda::wmma::load_matrix_sync(
          input_fragment, input_tile + input_begin * tensorcore_batch_tile, tensorcore_batch_tile);
      nvcuda::wmma::mma_sync(first_fragment, weight_fragment, input_fragment, first_fragment);
      __syncwarp();
    }}
    nvcuda::wmma::store_matrix_sync(
        matrix_tile, first_fragment, tensorcore_batch_tile, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < matrix_tile_elements; linear += 32) {{
      int const first_neuron = first_begin + linear / tensorcore_batch_tile;
      float activation_value = matrix_tile[linear];
      if (first_neuron < {first_size}) activation_value += {first_bias};
      activation_value = first_neuron < {first_size} ? {first_activation} : 0.0f;
      if (first_neuron < padded_first) {{
        first_tile[first_neuron * tensorcore_batch_tile + linear % tensorcore_batch_tile] =
            nvcuda::wmma::__float_to_tf32(activation_value);
      }}
    }}
    __syncwarp();
  }}

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> output_fragment;
  nvcuda::wmma::fill_fragment(output_fragment, 0.0f);
  for (int second_begin = 0; second_begin < {second_size}; second_begin += 16) {{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> second_fragment;
    nvcuda::wmma::fill_fragment(second_fragment, 0.0f);
    for (int first_begin = 0; first_begin < padded_first; first_begin += 8) {{
      for (int linear = lane; linear < weight_tile_elements; linear += 32) {{
        int const second_neuron = second_begin + linear / 8;
        int const first_neuron = first_begin + linear % 8;
        float const value = second_neuron < {second_size} && first_neuron < {first_size}
                                ? weights[{self.weight_offsets[second_weight]} +
                                          second_neuron * {first_size} + first_neuron] : 0.0f;
        weight_tile[linear] = nvcuda::wmma::__float_to_tf32(value);
      }}
      __syncwarp();
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          weight_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          first_fragment;
      nvcuda::wmma::load_matrix_sync(weight_fragment, weight_tile, 8);
      nvcuda::wmma::load_matrix_sync(
          first_fragment, first_tile + first_begin * tensorcore_batch_tile, tensorcore_batch_tile);
      nvcuda::wmma::mma_sync(second_fragment, weight_fragment, first_fragment, second_fragment);
      __syncwarp();
    }}
    nvcuda::wmma::store_matrix_sync(
        matrix_tile, second_fragment, tensorcore_batch_tile, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < matrix_tile_elements; linear += 32) {{
      int const second_neuron = second_begin + linear / tensorcore_batch_tile;
      float activation_value = matrix_tile[linear];
      if (second_neuron < {second_size}) activation_value += {second_bias};
      activation_value = second_neuron < {second_size} ? {second_activation} : 0.0f;
      matrix_tile[linear] = nvcuda::wmma::__float_to_tf32(activation_value);
    }}
    __syncwarp();

    for (int second_half = 0; second_half < 2; second_half++) {{
      for (int linear = lane; linear < weight_tile_elements; linear += 32) {{
        int const output_neuron = linear / 8;
        int const second_neuron = second_begin + second_half * 8 + linear % 8;
        float const value = output_neuron < {output_size} && second_neuron < {second_size}
                                ? weights[{self.weight_offsets[consumer_weight]} +
                                          output_neuron * {second_size} + second_neuron] : 0.0f;
        weight_tile[linear] = nvcuda::wmma::__float_to_tf32(value);
      }}
      __syncwarp();
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          weight_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,16,16,8,nvcuda::wmma::precision::tf32,nvcuda::wmma::row_major>
          second_fragment;
      nvcuda::wmma::load_matrix_sync(weight_fragment, weight_tile, 8);
      nvcuda::wmma::load_matrix_sync(
          second_fragment,
          matrix_tile + second_half * 8 * tensorcore_batch_tile,
          tensorcore_batch_tile);
      nvcuda::wmma::mma_sync(output_fragment, weight_fragment, second_fragment, output_fragment);
      __syncwarp();
    }}
  }}

  nvcuda::wmma::store_matrix_sync(
      matrix_tile, output_fragment, tensorcore_batch_tile, nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int linear = lane; linear < {output_size} * tensorcore_batch_tile; linear += 32) {{
    int const output_neuron = linear / tensorcore_batch_tile;
    int const local_batch = linear % tensorcore_batch_tile;
    int const ibatch = batch_begin + local_batch;
    if (ibatch < batch_size) {{
      matrix_tile[linear] += {consumer_bias};
      outputs[output_neuron * batch_size + ibatch] = {output_value};
    }}
  }}
}}
#endif"""

    def emit(self, output_path: Path, payload_elements: int, payload_scalar_code: int,
             streaming_output_threshold: int,
             explicit_half2_accumulators: dict[int, int] | None = None) -> None:
        num_inputs = self._size(self.graph.inputs[0])
        num_outputs = self._size(self.graph.outputs[0])
        body: list[str] = []
        team_body: list[str] = []
        dense_nodes = [node for node in self.graph.nodes if node.op in {"Dense", "DenseBiasActivation"}]
        none_accumulators = {node.id: 0 for node in dense_nodes}
        heuristic_accumulators = half2_accumulator_plan(
            self.graph, streaming_output_threshold, self.schedule
        )
        half_policies = [
            ("infer_batch_half2", none_accumulators),
            ("infer_batch_half2_heuristic", heuristic_accumulators),
        ]
        if explicit_half2_accumulators is not None:
            half_policies.append(("infer_batch_half2_explicit", explicit_half2_accumulators))

        output_id = self.graph.outputs[0]
        output_tensor = self.graph.tensors[output_id]
        source_output = output_tensor.producer is None
        if source_output and output_id != self.graph.inputs[0] and not output_tensor.is_constant:
            raise CompilerError(
                f"model output tensor {output_tensor.name!r} has no producer and is neither the input nor a constant"
            )

        def source_copy(read, write) -> list[str]:
            return [
                f"    for (int i = 0; i < {num_outputs}; i++) {{",
                f"      {write(output_id, 'i')} = {read(output_id, 'i')};",
                "    }",
            ]

        def build_half_body(accumulator_counts: dict[int, int]) -> list[str]:
            half_body: list[str] = []
            if source_output:
                half_body.extend(self._scope(source_copy(self._half_read, self._half_write)))
            for half_node in self.graph.nodes:
                if half_node.id in self.schedule.skipped_producers:
                    continue
                producer_id = self.schedule.pair_by_consumer.get(half_node.id)
                if producer_id is not None:
                    producer = self.graph.node_by_id(producer_id)
                    half_body.extend(self._scope(self._emit_half_streaming_dense_pair(
                        producer,
                        half_node,
                        accumulator_counts[producer.id],
                        accumulator_counts[half_node.id],
                    )))
                else:
                    count = accumulator_counts.get(half_node.id, 0)
                    half_body.extend(self._scope(self._emit_half_node(half_node, count)))
            return half_body

        if source_output:
            body.extend(self._scope(source_copy(self._read, self._write)))
            team_body.extend([
                f"          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, {num_outputs} * active_batch),",
                "                               [&](int linear) {",
                "            int const local_batch = linear % active_batch;",
                "            int const i = linear / active_batch;",
                "            int const ibatch = batch_begin + local_batch;",
                f"            {self._team_write(output_id, 'i')} = {self._team_read(output_id, 'i')};",
                "          });",
                "          team.team_barrier();",
            ])
        for node in self.graph.nodes:
            if node.id in self.schedule.skipped_producers:
                continue
            producer_id = self.schedule.pair_by_consumer.get(node.id)
            if producer_id is not None:
                producer = self.graph.node_by_id(producer_id)
                body.extend(self._scope(self._emit_streaming_dense_pair(producer, node, batch=False)))
            else:
                body.extend(self._scope(self._emit_node(node)))
        if not source_output:
            for node in self.graph.nodes:
                team_body.extend(self._emit_team_node(node))
        body_text = "\n".join(body)
        batch_body_text = "\n".join(f"      {line}" for line in body)
        half_body_texts = {
            method_name: "\n".join(f"      {line}" for line in build_half_body(accumulator_counts))
            for method_name, accumulator_counts in half_policies
        }
        team_body_text = "\n".join(team_body)
        local_workspace_elements = self.sample_plan.total_elements
        inline_workspace_declaration = (
            f"    Scalar workspace[{local_workspace_elements}];\n" if local_workspace_elements > 0 else ""
        )
        batch_workspace_declaration = (
            f"          Scalar workspace[{local_workspace_elements}];\n" if local_workspace_elements > 0 else ""
        )
        half_workspace_declaration = (
            f"          ponni::TwoHalf workspace[{local_workspace_elements}];\n"
            if local_workspace_elements > 0 else ""
        )
        batch_launch = f"""    InputView const input_view = inputs;
    OutputView const output_view = outputs;
    WeightView const weights = weights_;
    Kokkos::parallel_for(
        \"GeneratedModel::infer_batch\",
        Kokkos::RangePolicy<execution_space>(0, batch_size),
        KOKKOS_LAMBDA(int linear) {{
          int const ibatch = linear % batch_size;
          int const iwork = linear / batch_size;
          (void) iwork;
          ponni::SArray<Scalar,num_inputs> inputs;
          ponni::SArray<Scalar,num_outputs> outputs;
          for (int i = 0; i < num_inputs; i++) inputs(i) = input_view(i,ibatch);
          WeightView const weights_ = weights;
{batch_workspace_declaration}{batch_body_text}
          for (int i = 0; i < num_outputs; i++) output_view(i,ibatch) = outputs(i);
        }});"""
        def make_half_launch(method_name: str) -> str:
            return f"""    InputView const input_view = inputs;
    OutputView const output_view = outputs;
    HalfWeightView const half_weights = half_weights_;
    int const pair_count = (batch_size + 1) / 2;
    Kokkos::parallel_for(
        \"GeneratedModel::{method_name}\",
        Kokkos::RangePolicy<execution_space>(0, pair_count),
        KOKKOS_LAMBDA(int ipair) {{
          int const ibatch = 2 * ipair;
          bool const has_high_lane = ibatch + 1 < batch_size;
          ponni::SArray<ponni::TwoHalf,num_inputs> inputs;
          ponni::SArray<ponni::TwoHalf,num_outputs> outputs;
          for (int i = 0; i < num_inputs; i++) {{
            float const low = static_cast<float>(input_view(i,ibatch));
            float const high = has_high_lane ? static_cast<float>(input_view(i,ibatch + 1)) : 0.0f;
            inputs(i) = ponni::TwoHalf::from_floats(low, high);
          }}
{half_workspace_declaration}{half_body_texts[method_name]}
          for (int i = 0; i < num_outputs; i++) {{
            output_view(i,ibatch) = static_cast<Scalar>(outputs(i).low());
            if (has_high_lane) output_view(i,ibatch + 1) = static_cast<Scalar>(outputs(i).high());
          }}
        }});"""

        half_launches = {method_name: make_half_launch(method_name) for method_name, _ in half_policies}

        def make_half_method(method_name: str) -> str:
            return f"""  void {method_name}(InputView const & inputs, OutputView const & outputs) const {{
#ifndef NDEBUG
    if (!weights_loaded()) Kokkos::abort(\"GeneratedModel::{method_name} called before load_weights\");
    if (inputs.extent(0) != num_inputs) Kokkos::abort(\"GeneratedModel input feature extent is incorrect\");
    if (outputs.extent(0) != num_outputs) Kokkos::abort(\"GeneratedModel output feature extent is incorrect\");
    if (inputs.extent(1) != outputs.extent(1)) Kokkos::abort(\"GeneratedModel batch extents differ\");
#endif
    int const batch_size = checked_batch_size(inputs);
    if (batch_size == 0) return;
{half_launches[method_name]}
  }}"""

        half_methods = "\n\n".join(make_half_method(method_name) for method_name, _ in half_policies)
        tensorcore_chain = find_tensorcore_dense_chain(self.graph)
        tensorcore_scratch_bytes = estimate_tensorcore_scratch_bytes(self.graph, tensorcore_chain)
        tensorcore_eligible = (
            tensorcore_chain is not None and payload_scalar_code == 1 and num_outputs <= 16 and
            (len(tensorcore_chain) == 3 or num_inputs <= 8) and tensorcore_scratch_bytes <= 49152
        )
        tensorcore_hidden_size = self._size(tensorcore_chain[-2].outputs[0]) if tensorcore_chain is not None else 0
        if tensorcore_chain is not None and len(tensorcore_chain) == 3:
            maximum_tensorcore_warps = 1
            while (maximum_tensorcore_warps * 2 <= 8 and
                   maximum_tensorcore_warps * 2 * tensorcore_scratch_bytes <= 49152):
                maximum_tensorcore_warps *= 2
            if tensorcore_hidden_size <= 4:
                measured_tensorcore_warps = 4
            elif tensorcore_hidden_size <= 16:
                measured_tensorcore_warps = 2
            elif tensorcore_hidden_size <= 32:
                measured_tensorcore_warps = 4
            elif tensorcore_hidden_size <= 64:
                measured_tensorcore_warps = 2
            else:
                measured_tensorcore_warps = 1
            default_tensorcore_warps = min(maximum_tensorcore_warps, measured_tensorcore_warps)
        else:
            maximum_tensorcore_warps = 8
            default_tensorcore_warps = 4 if tensorcore_hidden_size <= 16 else (2 if tensorcore_hidden_size <= 256 else 1)
        if not tensorcore_eligible:
            tensorcore_kernel = ""
        elif len(tensorcore_chain) == 2:
            tensorcore_kernel = self._emit_tensorcore_dense_pair(*tensorcore_chain)
        else:
            tensorcore_kernel = self._emit_tensorcore_dense_triple(*tensorcore_chain)
        tensorcore_body = f"""#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
    static_assert(std::is_same_v<Scalar,float>, "Tensor Core inference requires Scalar=float");
    int const batch_size = checked_batch_size(inputs);
    if (batch_size == 0) return;
    if (warps_per_block < 1 || warps_per_block > maximum_tensorcore_warps_per_block) {{
      Kokkos::abort("GeneratedModel Tensor Core warps_per_block must be between 1 and 8");
    }}
    int const batch_tiles = (batch_size + tensorcore_batch_tile - 1) / tensorcore_batch_tile;
    int const block_count = (batch_tiles + warps_per_block - 1) / warps_per_block;
    int const thread_count = warps_per_block * 32;
    std::size_t const scratch_bytes = static_cast<std::size_t>(warps_per_block) *
                                      tensorcore_scratch_bytes_per_warp;
    Kokkos::Cuda const execution;
    {self.model_name}_tensorcore_kernel<<<block_count,thread_count,scratch_bytes,execution.cuda_stream()>>>(
        inputs.data(), outputs.data(), weights_.data(), batch_size);
#ifndef NDEBUG
    cudaError_t const launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) Kokkos::abort(cudaGetErrorString(launch_error));
#endif
#else
    (void) inputs;
    (void) outputs;
    (void) warps_per_block;
    Kokkos::abort("GeneratedModel Tensor Core inference requires CUDA Ampere or newer");
#endif""" if tensorcore_eligible else (
            "    (void) inputs;\n    (void) outputs;\n    (void) warps_per_block;\n"
            "    Kokkos::abort(\"GeneratedModel graph is not eligible for Tensor Core inference\");"
        )

        text = f"""#pragma once
// Generated deterministically by PONNI kokkos_nn. Do not edit.

#include \"ponni.h\"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
#include <cuda_runtime.h>
#include <mma.h>
#endif

namespace ponni::generated {{

{tensorcore_kernel}

template <class Scalar = float>
class {self.model_name} {{
public:
  int static constexpr num_inputs = {num_inputs};
  int static constexpr num_outputs = {num_outputs};
  int static constexpr workspace_elements = {self.plan.total_elements};
  int static constexpr sample_local_workspace_elements = {local_workspace_elements};
  int static constexpr default_hierarchical_batch_tile = {self.default_batch_tile};
  int static constexpr maximum_hierarchical_batch_tile = {self.maximum_batch_tile};
  int static constexpr tensorcore_batch_tile = 16;
  int static constexpr default_tensorcore_warps_per_block = {default_tensorcore_warps};
  int static constexpr maximum_tensorcore_warps_per_block = {maximum_tensorcore_warps};
  int static constexpr tensorcore_scratch_bytes_per_warp = {tensorcore_scratch_bytes};
  bool static constexpr tensorcore_eligible = {str(tensorcore_eligible).lower()};
  int static constexpr weight_elements = {payload_elements};
  using scalar_type = Scalar;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using InputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using OutputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using WeightView = Kokkos::View<Scalar*,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using HalfWeightView = Kokkos::View<Kokkos::Experimental::half_t*,Kokkos::LayoutRight,ponni::DeviceSpace>;

private:
  WeightView weights_;
  HalfWeightView half_weights_;

  static int checked_batch_size(InputView const & inputs) {{
    std::size_t const batch_size = inputs.extent(1);
    if (batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {{
      Kokkos::abort("GeneratedModel batch extent exceeds the generated 32-bit index range");
    }}
    return static_cast<int>(batch_size);
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_relu(Scalar value) {{
    return value > static_cast<Scalar>(0) ? value : static_cast<Scalar>(0);
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_sigmoid(Scalar value) {{
    if (value >= static_cast<Scalar>(0)) {{
      return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + Kokkos::exp(-value));
    }}
    Scalar const exp_value = Kokkos::exp(value);
    return exp_value / (static_cast<Scalar>(1) + exp_value);
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_tanh(Scalar value) {{ return Kokkos::tanh(value); }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_leaky_relu(Scalar value, Scalar alpha) {{
    return value >= static_cast<Scalar>(0) ? value : alpha * value;
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_elu(Scalar value, Scalar alpha) {{
    return value >= static_cast<Scalar>(0) ? value : alpha * (Kokkos::exp(value) - static_cast<Scalar>(1));
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_gelu(Scalar value, bool approximate) {{
    if (approximate) {{
      Scalar constexpr factor = static_cast<Scalar>(0.7978845608028654);
      return static_cast<Scalar>(0.5) * value *
             (static_cast<Scalar>(1) + Kokkos::tanh(factor *
              (value + static_cast<Scalar>(0.044715) * value * value * value)));
    }}
    return static_cast<Scalar>(0.5) * value *
           (static_cast<Scalar>(1) + Kokkos::erf(value * static_cast<Scalar>(0.7071067811865475)));
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_silu(Scalar value) {{ return value * apply_sigmoid(value); }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_softplus(Scalar value) {{
    Scalar const magnitude = Kokkos::abs(value);
    return (value > static_cast<Scalar>(0) ? value : static_cast<Scalar>(0)) +
           Kokkos::log(static_cast<Scalar>(1) + Kokkos::exp(-magnitude));
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_hard_sigmoid(Scalar value, Scalar alpha, Scalar beta) {{
    Scalar const transformed = alpha * value + beta;
    if (transformed < static_cast<Scalar>(0)) return static_cast<Scalar>(0);
    return transformed > static_cast<Scalar>(1) ? static_cast<Scalar>(1) : transformed;
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_hard_swish(Scalar value) {{
    return value * apply_hard_sigmoid(value, static_cast<Scalar>(1.0 / 6.0), static_cast<Scalar>(0.5));
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_mish(Scalar value) {{
    return value * Kokkos::tanh(apply_softplus(value));
  }}

  static std::uint64_t checksum(unsigned char const * data, std::size_t size) {{
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (std::size_t i = 0; i < size; i++) {{
      value ^= data[i];
      value *= UINT64_C(1099511628211);
    }}
    return value;
  }}

public:
  {self.model_name}() = default;

  bool weights_loaded() const {{ return weights_.is_allocated() && half_weights_.is_allocated(); }}

  bool load_weights(std::string const & path, std::string * error = nullptr) {{
    auto fail = [&](std::string const & message) {{
      if (error != nullptr) *error = message;
      return false;
    }};
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return fail(\"cannot open weight file: \" + path);
    std::streamsize const file_size = stream.tellg();
    if (file_size < 0) return fail(\"cannot determine weight-file size: \" + path);
    stream.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(file_size));
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), file_size)) return fail(\"cannot read weight file: \" + path);
    int constexpr header_size = 32;
    if (bytes.size() < header_size) return fail(\"weight file is shorter than its header\");
    unsigned char const expected_magic[8] = {{'P', 'N', 'N', 'W', 'G', 'T', '1', 0}};
    if (std::memcmp(bytes.data(), expected_magic, 8) != 0) return fail(\"invalid weight-file magic\");
    std::uint16_t const endian_probe = 1;
    if (*reinterpret_cast<unsigned char const *>(&endian_probe) != 1) return fail(\"little-endian host required\");
    std::uint32_t version = 0;
    std::uint32_t scalar_code = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t expected_checksum = 0;
    std::memcpy(&version, bytes.data() + 8, sizeof(version));
    std::memcpy(&scalar_code, bytes.data() + 12, sizeof(scalar_code));
    std::memcpy(&payload_bytes, bytes.data() + 16, sizeof(payload_bytes));
    std::memcpy(&expected_checksum, bytes.data() + 24, sizeof(expected_checksum));
    if (version != 1) return fail(\"unsupported weight-file version\");
    if (scalar_code != {payload_scalar_code}) return fail(\"weight-file scalar metadata does not match generated model\");
    if (bytes.size() != header_size + payload_bytes) return fail(\"weight-file payload size mismatch\");
    unsigned char const * payload = bytes.data() + header_size;
    if (checksum(payload, payload_bytes) != expected_checksum) return fail(\"weight-file checksum mismatch\");
    std::size_t constexpr stored_scalar_bytes = {4 if payload_scalar_code == 1 else 8};
    if (payload_bytes != static_cast<std::uint64_t>(weight_elements) * stored_scalar_bytes) {{
      return fail(\"weight-file element count does not match generated model\");
    }}
    Kokkos::View<Scalar*,Kokkos::LayoutRight,Kokkos::HostSpace> host_weights(\"generated_weights_host\", weight_elements);
    Kokkos::View<Kokkos::Experimental::half_t*,Kokkos::LayoutRight,Kokkos::HostSpace>
        host_half_weights(\"generated_half_weights_host\", weight_elements);
    for (int i = 0; i < weight_elements; i++) {{
      {'float' if payload_scalar_code == 1 else 'double'} stored_value;
      std::memcpy(&stored_value, payload + static_cast<std::size_t>(i) * stored_scalar_bytes, stored_scalar_bytes);
      host_weights(i) = static_cast<Scalar>(stored_value);
      host_half_weights(i) = Kokkos::Experimental::cast_to_half(static_cast<float>(stored_value));
    }}
    weights_ = WeightView(\"generated_weights\", weight_elements);
    half_weights_ = HalfWeightView(\"generated_half_weights\", weight_elements);
    Kokkos::deep_copy(weights_, host_weights);
    Kokkos::deep_copy(half_weights_, host_half_weights);
    return true;
  }}

  KOKKOS_INLINE_FUNCTION
  void infer_one(ponni::SArray<Scalar,num_inputs> const & inputs,
                 ponni::SArray<Scalar,num_outputs> & outputs) const {{
{inline_workspace_declaration}{body_text}
  }}

  void infer_batch(InputView const & inputs, OutputView const & outputs) const {{
#ifndef NDEBUG
    if (!weights_loaded()) Kokkos::abort(\"GeneratedModel::infer_batch called before load_weights\");
    if (inputs.extent(0) != num_inputs) Kokkos::abort(\"GeneratedModel input feature extent is incorrect\");
    if (outputs.extent(0) != num_outputs) Kokkos::abort(\"GeneratedModel output feature extent is incorrect\");
    if (inputs.extent(1) != outputs.extent(1)) Kokkos::abort(\"GeneratedModel batch extents differ\");
#endif
    int const batch_size = checked_batch_size(inputs);
{batch_launch}
  }}

{half_methods}

  void infer_batch_hierarchical(InputView const & inputs, OutputView const & outputs,
                                int batch_tile = default_hierarchical_batch_tile) const {{
#ifndef NDEBUG
    if (!weights_loaded()) Kokkos::abort("GeneratedModel::infer_batch_hierarchical called before load_weights");
    if (inputs.extent(0) != num_inputs) Kokkos::abort("GeneratedModel input feature extent is incorrect");
    if (outputs.extent(0) != num_outputs) Kokkos::abort("GeneratedModel output feature extent is incorrect");
    if (inputs.extent(1) != outputs.extent(1)) Kokkos::abort("GeneratedModel batch extents differ");
#endif
    if (batch_tile < 1 || batch_tile > maximum_hierarchical_batch_tile) {{
      Kokkos::abort("GeneratedModel hierarchical batch tile is outside the generated limits");
    }}
    int const batch_size = checked_batch_size(inputs);
    if (batch_size == 0) return;
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;
    int const league_size = (batch_size + batch_tile - 1) / batch_tile;
    int const scratch_bytes = workspace_elements * batch_tile * static_cast<int>(sizeof(Scalar));
    policy_type policy(league_size, Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    WeightView const weights = weights_;
    Kokkos::parallel_for(
        "GeneratedModel::infer_batch_hierarchical", policy,
        KOKKOS_LAMBDA(member_type const & team) {{
          int const batch_begin = team.league_rank() * batch_tile;
          int const remaining_batch = batch_size - batch_begin;
          int const active_batch = remaining_batch < batch_tile ? remaining_batch : batch_tile;
          Scalar * workspace = nullptr;
          if (scratch_bytes > 0) {{
            workspace = reinterpret_cast<Scalar *>(team.team_shmem().get_shmem(scratch_bytes));
          }}
{team_body_text}
        }});
  }}

  void infer_batch_tensorcore(
      InputView const & inputs, OutputView const & outputs,
      int warps_per_block = default_tensorcore_warps_per_block) const {{
#ifndef NDEBUG
    if (!weights_loaded()) Kokkos::abort("GeneratedModel::infer_batch_tensorcore called before load_weights");
    if (inputs.extent(0) != num_inputs) Kokkos::abort("GeneratedModel input feature extent is incorrect");
    if (outputs.extent(0) != num_outputs) Kokkos::abort("GeneratedModel output feature extent is incorrect");
    if (inputs.extent(1) != outputs.extent(1)) Kokkos::abort("GeneratedModel batch extents differ");
#endif
{tensorcore_body}
  }}
}};

}}  // namespace ponni::generated
"""
        output_path.write_text(text)


def emit_cpp(graph: Graph, plan: StoragePlan, sample_plan: StoragePlan, schedule: DenseChainSchedule,
             offsets: dict[int, int], output_dir: Path, model_name: str,
             strategy: str, payload_elements: int, payload_scalar_code: int,
             default_batch_tile: int, maximum_batch_tile: int, streaming_output_threshold: int,
             explicit_half2_accumulators: dict[int, int] | None = None) -> Path:
    output_path = output_dir / f"{model_name}.hpp"
    CppEmitter(
        graph, plan, sample_plan, schedule, offsets, model_name, strategy,
        default_batch_tile, maximum_batch_tile
    ).emit(
        output_path, payload_elements, payload_scalar_code, streaming_output_threshold,
        explicit_half2_accumulators
    )
    return output_path
