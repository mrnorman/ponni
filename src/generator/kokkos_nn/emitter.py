from __future__ import annotations

from pathlib import Path
import re

from .errors import CompilerError
from .ir import DType, Graph, Node
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
    """Select the conservative cross-vendor half2 accumulation policy."""
    del dot_length, simultaneous_outputs
    # NVIDIA Ampere benefited from multiple partials at long dot lengths, but
    # optimized MI250X measurements favored one dependency chain at every
    # measured width. Keep the lower-resource cross-vendor choice by default;
    # explicit policies remain available when accuracy or one target warrants it.
    return 0


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


class CppEmitter:
    def __init__(self, graph: Graph, plan: StoragePlan, sample_plan: StoragePlan,
                 mask_plan: StoragePlan, sample_mask_plan: StoragePlan,
                 schedule: DenseChainSchedule, weight_offsets: dict[int, int], model_name: str,
                 strategy: str, default_batch_tile: int, maximum_batch_tile: int) -> None:
        if strategy not in {"sample-local", "team", "half2"}:
            raise CompilerError(
                f"unknown execution strategy {strategy!r}; choose sample-local, team, half2, or auto"
            )
        self.graph = graph
        self.plan = plan
        self.sample_plan = sample_plan
        self.mask_plan = mask_plan
        self.sample_mask_plan = sample_mask_plan
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
        if tensor.dtype == DType.BOOL:
            return self._mask_read(tensor_id, index)
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            if tensor_id not in self.weight_offsets:
                raise CompilerError(f"no emitted weight offset for constant tensor {tensor.name!r}")
            return f"parameters_({self.weight_offsets[tensor_id]} + {use_index})"
        if tensor_id == self.graph.inputs[0]:
            return f"inputs({use_index})"
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({use_index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no activation storage assigned to tensor {tensor.name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {use_index}]"

    def _write(self, tensor_id: int, index: str) -> str:
        if self.graph.tensors[tensor_id].dtype == DType.BOOL:
            return self._mask_write(tensor_id, index)
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(f"no activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}")
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    def _batch_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        if tensor.dtype == DType.BOOL:
            return self._batch_mask_read(tensor_id, index)
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
        if self.graph.tensors[tensor_id].dtype == DType.BOOL:
            return self._batch_mask_write(tensor_id, index)
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index},ibatch)"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(
                f"no batch-local activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    def _team_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        if tensor.dtype == DType.BOOL:
            return self._team_mask_read(tensor_id, index)
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
        if self.graph.tensors[tensor_id].dtype == DType.BOOL:
            return self._team_mask_write(tensor_id, index)
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index},ibatch)"
        if tensor_id not in self.plan.slots:
            raise CompilerError(
                f"no hierarchical scratch storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[({self.plan.slots[tensor_id].offset} + {index}) * batch_tile + local_batch]"

    def _half_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        if tensor.dtype == DType.BOOL:
            return self._half_mask_read(tensor_id, index)
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
        if self.graph.tensors[tensor_id].dtype == DType.BOOL:
            return self._half_mask_write(tensor_id, index)
        if tensor_id == self.graph.outputs[0]:
            return f"outputs({index})"
        if tensor_id not in self.sample_plan.slots:
            raise CompilerError(
                f"no half2 activation storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"workspace[{self.sample_plan.slots[tensor_id].offset} + {index}]"

    def _mask_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            return f"(parameters_({self.weight_offsets[tensor_id]} + {use_index}) != static_cast<Scalar>(0))"
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(f"no mask storage assigned to tensor {tensor.name!r}")
        return f"(mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {use_index}] != 0)"

    def _mask_write(self, tensor_id: int, index: str) -> str:
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(f"no mask storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}")
        return f"mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {index}]"

    def _batch_mask_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            return f"(weights({self.weight_offsets[tensor_id]} + {use_index}) != static_cast<Scalar>(0))"
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(f"no batch-local mask storage assigned to tensor {tensor.name!r}")
        return f"(mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {use_index}] != 0)"

    def _batch_mask_write(self, tensor_id: int, index: str) -> str:
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(
                f"no batch-local mask storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {index}]"

    def _team_mask_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            return f"(weights({self.weight_offsets[tensor_id]} + {use_index}) != static_cast<Scalar>(0))"
        if tensor_id not in self.mask_plan.slots:
            raise CompilerError(f"no hierarchical mask scratch assigned to tensor {tensor.name!r}")
        return (
            f"(mask_workspace[({self.mask_plan.slots[tensor_id].offset} + {use_index}) * "
            "batch_tile + local_batch] != 0)"
        )

    def _team_mask_write(self, tensor_id: int, index: str) -> str:
        if tensor_id not in self.mask_plan.slots:
            raise CompilerError(
                f"no hierarchical mask scratch assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"mask_workspace[({self.mask_plan.slots[tensor_id].offset} + {index}) * batch_tile + local_batch]"

    def _half_mask_read(self, tensor_id: int, index: str) -> str:
        tensor = self.graph.tensors[tensor_id]
        use_index = "0" if tensor.sample_size == 1 else index
        if tensor.is_constant:
            return (
                "ponni::TwoMask::splat(static_cast<float>(half_weights("
                f"{self.weight_offsets[tensor_id]} + {use_index})) != 0.0f)"
            )
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(f"no half2 mask storage assigned to tensor {tensor.name!r}")
        return f"mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {use_index}]"

    def _half_mask_write(self, tensor_id: int, index: str) -> str:
        if tensor_id not in self.sample_mask_plan.slots:
            raise CompilerError(
                f"no half2 mask storage assigned to output tensor {self.graph.tensors[tensor_id].name!r}"
            )
        return f"mask_workspace[{self.sample_mask_plan.slots[tensor_id].offset} + {index}]"

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
        attributes = attributes or {}
        if name == "Celu":
            alpha = float(attributes.get("alpha", 1.0))
            return (
                f"({expression} > static_cast<Scalar>(0) ? {expression} : static_cast<Scalar>({alpha!r}) * "
                f"(Kokkos::exp({expression} / static_cast<Scalar>({alpha!r})) - static_cast<Scalar>(1)))"
            )
        if name == "Selu":
            alpha = float(attributes.get("alpha", 1.6732631921768188))
            gamma = float(attributes.get("gamma", 1.0507010221481323))
            return (
                f"(static_cast<Scalar>({gamma!r}) * ({expression} > static_cast<Scalar>(0) ? {expression} : "
                f"static_cast<Scalar>({alpha!r}) * (Kokkos::exp({expression}) - static_cast<Scalar>(1))))"
            )
        if name == "Softsign":
            return f"({expression} / (static_cast<Scalar>(1) + Kokkos::abs({expression})))"
        if name == "ThresholdedRelu":
            alpha = float(attributes.get("alpha", 1.0))
            return f"({expression} > static_cast<Scalar>({alpha!r}) ? {expression} : static_cast<Scalar>(0))"
        function = {
            "Abs": "Kokkos::abs", "Acos": "Kokkos::acos", "Acosh": "Kokkos::acosh", "Asin": "Kokkos::asin",
            "Asinh": "Kokkos::asinh", "Atan": "Kokkos::atan", "Atanh": "Kokkos::atanh", "Ceil": "Kokkos::ceil",
            "Cos": "Kokkos::cos", "Cosh": "Kokkos::cosh", "Erf": "Kokkos::erf", "Exp": "Kokkos::exp",
            "Floor": "Kokkos::floor", "Log": "Kokkos::log", "Sin": "Kokkos::sin", "Sinh": "Kokkos::sinh",
            "Sqrt": "Kokkos::sqrt", "Tan": "Kokkos::tan",
        }.get(name)
        if name == "Neg":
            return f"(-{expression})"
        if name == "Reciprocal":
            return f"(static_cast<Scalar>(1) / {expression})"
        if name == "Round":
            return f"apply_round({expression})"
        if name == "Sign":
            return f"apply_sign({expression})"
        if function is None:
            raise CompilerError(f"C++ emitter has no unary implementation for {name}")
        return f"{function}({expression})"

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

    @staticmethod
    def _comparison(op: str, left: str, right: str, half: bool = False,
                    boolean_inputs: bool = False) -> str:
        if half:
            if boolean_inputs:
                if op != "Equal":
                    raise CompilerError(f"unsupported Boolean comparison {op}")
                return f"ponni::TwoMask::equal({left}, {right})"
            function = {
                "Equal": "equal", "Greater": "greater", "GreaterOrEqual": "greater_or_equal", "Less": "less",
                "LessOrEqual": "less_or_equal",
            }[op]
            return f"ponni::TwoHalf::{function}({left}, {right})"
        symbol = {"Equal": "==", "Greater": ">", "GreaterOrEqual": ">=", "Less": "<", "LessOrEqual": "<="}[op]
        return f"({left} {symbol} {right})"

    @staticmethod
    def _logical(op: str, left: str, right: str | None = None, half: bool = False) -> str:
        if half:
            function = {"And": "logical_and", "Not": "logical_not", "Or": "logical_or", "Xor": "logical_xor"}[op]
            arguments = left if right is None else f"{left}, {right}"
            return f"ponni::TwoMask::{function}({arguments})"
        if op == "Not":
            return f"(!{left})"
        symbol = {"And": "&&", "Or": "||", "Xor": "!="}[op]
        return f"({left} {symbol} {right})"

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
        if name == "Celu":
            return f"ponni::TwoHalf::celu({expression}, {float(attributes.get('alpha', 1.0))!r}f)"
        if name == "Selu":
            alpha = float(attributes.get("alpha", 1.6732631921768188))
            gamma = float(attributes.get("gamma", 1.0507010221481323))
            return f"ponni::TwoHalf::selu({expression}, {alpha!r}f, {gamma!r}f)"
        if name == "ThresholdedRelu":
            return (
                f"ponni::TwoHalf::thresholded_relu({expression}, "
                f"{float(attributes.get('alpha', 1.0))!r}f)"
            )
        if name == "Gelu":
            approximate = str(attributes.get("approximate", "none")) == "tanh"
            return f"ponni::TwoHalf::gelu({expression}, {str(approximate).lower()})"
        if name == "HardSigmoid":
            alpha = float(attributes.get("alpha", 0.2))
            beta = float(attributes.get("beta", 0.5))
            return f"ponni::TwoHalf::hard_sigmoid({expression}, {alpha!r}f, {beta!r}f)"
        function = {
            "Abs": "abs", "Acos": "acos", "Acosh": "acosh", "Asin": "asin", "Asinh": "asinh", "Atan": "atan",
            "Atanh": "atanh", "Ceil": "ceil", "Cos": "cos", "Cosh": "cosh", "Erf": "erf", "Exp": "exp",
            "Floor": "floor", "HardSwish": "hard_swish", "Log": "log", "Mish": "mish", "Reciprocal": "reciprocal",
            "Relu": "relu", "Round": "round", "Sigmoid": "sigmoid", "Sign": "sign", "Silu": "silu", "Sin": "sin",
            "Sinh": "sinh", "Softplus": "softplus", "Softsign": "softsign", "Sqrt": "sqrt", "Tan": "tan",
            "Tanh": "tanh",
        }.get(name)
        if function is None:
            raise CompilerError(f"half2 C++ emitter has no unary implementation for {name}")
        return f"ponni::TwoHalf::{function}({expression})"

    def _scalar_reduction(self, node: Node, read, write, indent: str) -> list[str]:
        input_id = node.inputs[0]
        input_size = self._size(input_id)
        value = lambda index: read(input_id, index)
        lines: list[str] = []
        if node.op in {"ReduceMax", "ReduceMin"}:
            lines.append(f"{indent}Scalar reduction = {value('0')};")
            comparison = ">" if node.op == "ReduceMax" else "<"
            lines.append(f"{indent}for (int i = 1; i < {input_size}; i++) {{")
            lines.append(f"{indent}  Scalar const value = {value('i')};")
            lines.append(f"{indent}  reduction = value {comparison} reduction ? value : reduction;")
            lines.append(f"{indent}}}")
        elif node.op == "ReduceLogSumExp":
            lines.append(f"{indent}Scalar maximum = {value('0')};")
            lines.append(f"{indent}for (int i = 1; i < {input_size}; i++) {{")
            lines.append(f"{indent}  Scalar const value = {value('i')};")
            lines.append(f"{indent}  maximum = value > maximum ? value : maximum;")
            lines.append(f"{indent}}}")
            lines.append(f"{indent}Scalar reduction = static_cast<Scalar>(0);")
            lines.append(
                f"{indent}for (int i = 0; i < {input_size}; i++) reduction += Kokkos::exp({value('i')} - maximum);"
            )
            lines.append(
                f"{indent}reduction = Kokkos::isinf(maximum) ? maximum : maximum + Kokkos::log(reduction);"
            )
        else:
            initial = "static_cast<Scalar>(1)" if node.op == "ReduceProd" else "static_cast<Scalar>(0)"
            lines.append(f"{indent}Scalar reduction = {initial};")
            expression = value("i")
            if node.op == "ReduceL1":
                expression = f"Kokkos::abs({expression})"
            elif node.op in {"ReduceL2", "ReduceSumSquare"}:
                expression = f"({expression} * {expression})"
            operator = "*=" if node.op == "ReduceProd" else "+="
            lines.append(f"{indent}for (int i = 0; i < {input_size}; i++) reduction {operator} {expression};")
            if node.op == "ReduceMean":
                lines.append(f"{indent}reduction /= static_cast<Scalar>({input_size});")
            elif node.op == "ReduceL2":
                lines.append(f"{indent}reduction = Kokkos::sqrt(reduction);")
            elif node.op == "ReduceLogSum":
                lines.append(f"{indent}reduction = Kokkos::log(reduction);")
        lines.append(f"{indent}{write(node.outputs[0], '0')} = reduction;")
        return lines

    def _half_reduction(self, node: Node) -> list[str]:
        input_id = node.inputs[0]
        input_size = self._size(input_id)
        value = lambda index: self._half_read(input_id, index)
        lines: list[str] = []
        if node.op in {"ReduceMax", "ReduceMin"}:
            function = "maximum" if node.op == "ReduceMax" else "minimum"
            lines.append(f"    ponni::TwoHalf reduction = {value('0')};")
            lines.append(
                f"    for (int i = 1; i < {input_size}; i++) reduction = "
                f"ponni::TwoHalf::{function}(reduction, {value('i')});"
            )
        elif node.op == "ReduceLogSumExp":
            lines.append(f"    ponni::TwoHalf maximum = {value('0')};")
            lines.append(
                f"    for (int i = 1; i < {input_size}; i++) maximum = "
                f"ponni::TwoHalf::maximum(maximum, {value('i')});"
            )
            lines.append("    ponni::TwoHalf reduction = ponni::TwoHalf::zero();")
            lines.append(
                f"    for (int i = 0; i < {input_size}; i++) reduction = reduction + "
                f"ponni::TwoHalf::exp({value('i')} - maximum);"
            )
            lines.append(
                "    reduction = ponni::TwoHalf::select(ponni::TwoHalf::is_inf(maximum, true, true), maximum, "
                "maximum + ponni::TwoHalf::log(reduction));"
            )
        else:
            initial = "ponni::TwoHalf::from_floats(1.0f, 1.0f)" if node.op == "ReduceProd" else "ponni::TwoHalf::zero()"
            lines.append(f"    ponni::TwoHalf reduction = {initial};")
            expression = value("i")
            if node.op == "ReduceL1":
                expression = f"ponni::TwoHalf::abs({expression})"
            elif node.op in {"ReduceL2", "ReduceSumSquare"}:
                expression = f"({expression} * {expression})"
            operator = "*" if node.op == "ReduceProd" else "+"
            lines.append(
                f"    for (int i = 0; i < {input_size}; i++) reduction = reduction {operator} {expression};"
            )
            if node.op == "ReduceMean":
                lines.append(
                    f"    reduction = reduction / ponni::TwoHalf::from_floats({input_size}.0f, {input_size}.0f);"
                )
            elif node.op == "ReduceL2":
                lines.append("    reduction = ponni::TwoHalf::sqrt(reduction);")
            elif node.op == "ReduceLogSum":
                lines.append("    reduction = ponni::TwoHalf::log(reduction);")
        lines.append(f"    {self._half_write(node.outputs[0], '0')} = reduction;")
        return lines

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
        if node.op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._comparison(
                node.op, read(node.inputs[0], "i"), read(node.inputs[1], "i"),
                boolean_inputs=self.graph.tensors[node.inputs[0]].dtype == DType.BOOL,
            )
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"And", "Or", "Xor"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._logical(node.op, read(node.inputs[0], "i"), read(node.inputs[1], "i"))
            lines.append(f"      {write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "Not":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      {write(output_id, 'i')} = {self._logical('Not', read(node.inputs[0], 'i'))};")
            lines.append("    }")
            return lines
        if node.op == "Cast":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(f"      {write(output_id, 'i')} = static_cast<Scalar>({read(node.inputs[0], 'i')});")
            lines.append("    }")
            return lines
        if node.op == "PRelu":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = read(node.inputs[0], "i")
            slope = read(node.inputs[1], "i")
            lines.append(f"      {write(output_id, 'i')} = {value} >= static_cast<Scalar>(0) ? {value} : {value} * {slope};")
            lines.append("    }")
            return lines
        if node.op in {"Mean", "Sum"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            expression = " + ".join(read(tensor_id, "i") for tensor_id in node.inputs)
            if node.op == "Mean":
                expression = f"({expression}) / static_cast<Scalar>({len(node.inputs)})"
            lines.append(f"      {write(output_id, 'i')} = {expression};")
            lines.append("    }")
            return lines
        if node.op in {"IsInf", "IsNaN"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = read(node.inputs[0], "i")
            if node.op == "IsNaN":
                predicate = f"Kokkos::isnan({value})"
            else:
                negative = bool(int(node.attributes.get("detect_negative", 1)))
                positive = bool(int(node.attributes.get("detect_positive", 1)))
                signs = []
                if negative:
                    signs.append(f"{value} < static_cast<Scalar>(0)")
                if positive:
                    signs.append(f"{value} > static_cast<Scalar>(0)")
                predicate = f"(Kokkos::isinf({value}) && ({' || '.join(signs) if signs else 'false'}))"
            lines.append(f"      {write(output_id, 'i')} = {predicate};")
            lines.append("    }")
            return lines
        if node.op in {"Where", "CompareSelect"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            if node.op == "Where":
                condition = read(node.inputs[0], "i")
                when_true = read(node.inputs[1], "i")
                when_false = read(node.inputs[2], "i")
            else:
                condition = self._comparison(
                    str(node.attributes["comparison"]), read(node.inputs[0], "i"), read(node.inputs[1], "i")
                )
                when_true = read(node.inputs[2], "i")
                when_false = read(node.inputs[3], "i")
            lines.append(f"      {write(output_id, 'i')} = {condition} ? {when_true} : {when_false};")
            lines.append("    }")
            return lines
        if node.op in {"Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos", "Cosh",
                        "Elu", "Erf", "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                        "Neg", "Reciprocal", "Relu", "Round", "Selu", "Sigmoid", "Sign", "Silu", "Sin", "Sinh",
                        "Softplus", "Softsign", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"}:
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
        if node.op.startswith("Reduce"):
            return self._scalar_reduction(node, read, write, "    ")
        if node.op == "LpNormalization":
            input_id = node.inputs[0]
            p = int(node.attributes.get("p", 2))
            lines.append("    Scalar norm = static_cast<Scalar>(0);")
            expression = f"Kokkos::abs({read(input_id, 'i')})"
            if p == 2:
                expression = f"({read(input_id, 'i')} * {read(input_id, 'i')})"
            lines.append(f"    for (int i = 0; i < {output_size}; i++) norm += {expression};")
            if p == 2:
                lines.append("    norm = Kokkos::sqrt(norm);")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            lines.append(
                f"      {write(output_id, 'i')} = norm == static_cast<Scalar>(0) ? static_cast<Scalar>(0) : "
                f"{read(input_id, 'i')} / norm;"
            )
            lines.append("    }")
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
        if node.op == "Gather":
            for output_index, input_index in enumerate(node.attributes["indices"]):
                lines.append(f"    {write(output_id, str(output_index))} = {read(node.inputs[0], str(input_index))};")
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
        if node.op in {"LayerNormalization", "LogSoftmax", "LpNormalization", "Softmax"} or node.op.startswith("Reduce"):
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
            elif node.op == "LpNormalization":
                input_id = node.inputs[0]
                p = int(node.attributes.get("p", 2))
                lines.append("            Scalar norm = static_cast<Scalar>(0);")
                expression = f"Kokkos::abs({self._team_read(input_id, 'i')})"
                if p == 2:
                    value = self._team_read(input_id, "i")
                    expression = f"({value} * {value})"
                lines.append(f"            for (int i = 0; i < {output_size}; i++) norm += {expression};")
                if p == 2:
                    lines.append("            norm = Kokkos::sqrt(norm);")
                lines.append(f"            for (int i = 0; i < {output_size}; i++) {{")
                lines.append(
                    f"              {self._team_write(output_id, 'i')} = norm == static_cast<Scalar>(0) ? "
                    f"static_cast<Scalar>(0) : {self._team_read(input_id, 'i')} / norm;"
                )
                lines.append("            }")
            else:
                lines.extend(self._scalar_reduction(node, self._team_read, self._team_write, "            "))
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
        elif node.op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
            value = self._comparison(
                node.op, self._team_read(node.inputs[0], "i"), self._team_read(node.inputs[1], "i"),
                boolean_inputs=self.graph.tensors[node.inputs[0]].dtype == DType.BOOL,
            )
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op in {"And", "Or", "Xor"}:
            value = self._logical(
                node.op, self._team_read(node.inputs[0], "i"), self._team_read(node.inputs[1], "i")
            )
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "Not":
            value = self._logical("Not", self._team_read(node.inputs[0], "i"))
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "Cast":
            value = f"static_cast<Scalar>({self._team_read(node.inputs[0], 'i')})"
            lines.append(f"            {self._team_write(output_id, 'i')} = {value};")
        elif node.op == "PRelu":
            value = self._team_read(node.inputs[0], "i")
            slope = self._team_read(node.inputs[1], "i")
            lines.append(
                f"            {self._team_write(output_id, 'i')} = {value} >= static_cast<Scalar>(0) ? "
                f"{value} : {value} * {slope};"
            )
        elif node.op in {"Mean", "Sum"}:
            expression = " + ".join(self._team_read(tensor_id, "i") for tensor_id in node.inputs)
            if node.op == "Mean":
                expression = f"({expression}) / static_cast<Scalar>({len(node.inputs)})"
            lines.append(f"            {self._team_write(output_id, 'i')} = {expression};")
        elif node.op in {"IsInf", "IsNaN"}:
            value = self._team_read(node.inputs[0], "i")
            if node.op == "IsNaN":
                predicate = f"Kokkos::isnan({value})"
            else:
                negative = bool(int(node.attributes.get("detect_negative", 1)))
                positive = bool(int(node.attributes.get("detect_positive", 1)))
                signs = []
                if negative:
                    signs.append(f"{value} < static_cast<Scalar>(0)")
                if positive:
                    signs.append(f"{value} > static_cast<Scalar>(0)")
                predicate = f"(Kokkos::isinf({value}) && ({' || '.join(signs) if signs else 'false'}))"
            lines.append(f"            {self._team_write(output_id, 'i')} = {predicate};")
        elif node.op in {"Where", "CompareSelect"}:
            if node.op == "Where":
                condition = self._team_read(node.inputs[0], "i")
                when_true = self._team_read(node.inputs[1], "i")
                when_false = self._team_read(node.inputs[2], "i")
            else:
                condition = self._comparison(
                    str(node.attributes["comparison"]), self._team_read(node.inputs[0], "i"),
                    self._team_read(node.inputs[1], "i"),
                )
                when_true = self._team_read(node.inputs[2], "i")
                when_false = self._team_read(node.inputs[3], "i")
            lines.append(
                f"            {self._team_write(output_id, 'i')} = {condition} ? {when_true} : {when_false};"
            )
        elif node.op in {"Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos",
                            "Cosh", "Elu", "Erf", "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu",
                            "Log", "Mish", "Neg", "Reciprocal", "Relu", "Round", "Selu", "Sigmoid", "Sign", "Silu",
                            "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"}:
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
        elif node.op == "Gather":
            indices = node.attributes["indices"]
            lines.append(f"            switch (i) {{")
            for output_index, input_index in enumerate(indices):
                lines.append(
                    f"              case {output_index}: {self._team_write(output_id, 'i')} = "
                    f"{self._team_read(node.inputs[0], str(input_index))}; break;"
                )
            lines.append("            }")
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
        if node.op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._comparison(
                node.op, self._half_read(node.inputs[0], "i"), self._half_read(node.inputs[1], "i"), half=True,
                boolean_inputs=self.graph.tensors[node.inputs[0]].dtype == DType.BOOL,
            )
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"And", "Or", "Xor"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._logical(
                node.op, self._half_read(node.inputs[0], "i"), self._half_read(node.inputs[1], "i"), half=True
            )
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "Not":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._logical("Not", self._half_read(node.inputs[0], "i"), half=True)
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "Cast":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = (
                f"ponni::TwoHalf::select({self._half_read(node.inputs[0], 'i')}, "
                "ponni::TwoHalf::from_floats(1.0f, 1.0f), ponni::TwoHalf::zero())"
            )
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op == "PRelu":
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._half_read(node.inputs[0], "i")
            slope = self._half_read(node.inputs[1], "i")
            lines.append(f"      {self._half_write(output_id, 'i')} = ponni::TwoHalf::prelu({value}, {slope});")
            lines.append("    }")
            return lines
        if node.op in {"Mean", "Sum"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            expression = " + ".join(self._half_read(tensor_id, "i") for tensor_id in node.inputs)
            if node.op == "Mean":
                divisor = len(node.inputs)
                expression = f"({expression}) / ponni::TwoHalf::from_floats({divisor}.0f, {divisor}.0f)"
            lines.append(f"      {self._half_write(output_id, 'i')} = {expression};")
            lines.append("    }")
            return lines
        if node.op in {"IsInf", "IsNaN"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = self._half_read(node.inputs[0], "i")
            if node.op == "IsNaN":
                predicate = f"ponni::TwoHalf::is_nan({value})"
            else:
                negative = str(bool(int(node.attributes.get("detect_negative", 1)))).lower()
                positive = str(bool(int(node.attributes.get("detect_positive", 1)))).lower()
                predicate = f"ponni::TwoHalf::is_inf({value}, {negative}, {positive})"
            lines.append(f"      {self._half_write(output_id, 'i')} = {predicate};")
            lines.append("    }")
            return lines
        if node.op in {"Where", "CompareSelect"}:
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            if node.op == "Where":
                condition = self._half_read(node.inputs[0], "i")
                when_true = self._half_read(node.inputs[1], "i")
                when_false = self._half_read(node.inputs[2], "i")
            else:
                condition = self._comparison(
                    str(node.attributes["comparison"]), self._half_read(node.inputs[0], "i"),
                    self._half_read(node.inputs[1], "i"), half=True,
                )
                when_true = self._half_read(node.inputs[2], "i")
                when_false = self._half_read(node.inputs[3], "i")
            value = f"ponni::TwoHalf::select({condition}, {when_true}, {when_false})"
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
            return lines
        if node.op in {"Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos", "Cosh",
                        "Elu", "Erf", "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish",
                        "Neg", "Reciprocal", "Relu", "Round", "Selu", "Sigmoid", "Sign", "Silu", "Sin", "Sinh",
                        "Softplus", "Softsign", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"}:
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
        if node.op.startswith("Reduce"):
            return self._half_reduction(node)
        if node.op == "LpNormalization":
            input_id = node.inputs[0]
            p = int(node.attributes.get("p", 2))
            lines.append("    ponni::TwoHalf norm = ponni::TwoHalf::zero();")
            expression = f"ponni::TwoHalf::abs({self._half_read(input_id, 'i')})"
            if p == 2:
                value = self._half_read(input_id, "i")
                expression = f"({value} * {value})"
            lines.append(f"    for (int i = 0; i < {output_size}; i++) norm = norm + {expression};")
            if p == 2:
                lines.append("    norm = ponni::TwoHalf::sqrt(norm);")
            lines.append("    ponni::TwoMask const zero_norm = ponni::TwoHalf::equal(norm, ponni::TwoHalf::zero());")
            lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
            value = (
                f"ponni::TwoHalf::select(zero_norm, ponni::TwoHalf::zero(), "
                f"{self._half_read(input_id, 'i')} / norm)"
            )
            lines.append(f"      {self._half_write(output_id, 'i')} = {value};")
            lines.append("    }")
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
        if node.op == "Gather":
            for output_index, input_index in enumerate(node.attributes["indices"]):
                lines.append(
                    f"    {self._half_write(output_id, str(output_index))} = "
                    f"{self._half_read(node.inputs[0], str(input_index))};"
                )
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

    def emit(self, output_path: Path, payload_elements: int, payload_scalar_code: int,
             streaming_output_threshold: int,
             explicit_half2_accumulators: dict[int, int] | None = None,
             emit_half2_heuristic: bool = False) -> None:
        num_inputs = self._size(self.graph.inputs[0])
        num_outputs = self._size(self.graph.outputs[0])
        learned_tensors = [
            tensor for _, tensor in sorted(self.graph.tensors.items())
            if tensor.is_constant and tensor.constant_name is not None and
            self.graph.constants[tensor.constant_name].learned
        ]
        learned_ranges: list[tuple[int, int, int]] = []
        learned_offset = 0
        for tensor in learned_tensors:
            learned_ranges.append((learned_offset, self.weight_offsets[tensor.id], tensor.sample_size))
            learned_offset += tensor.sample_size
        parameter_index_lines = []
        for parameter_begin, storage_begin, size in learned_ranges:
            parameter_end = parameter_begin + size
            parameter_index_lines.append(
                f"    if (index < {parameter_end}) return {storage_begin} + index - {parameter_begin};"
            )
        parameter_index_lines.extend([
            '    Kokkos::abort("GeneratedModel learned parameter index is out of range");',
            "    return 0;",
        ])
        parameter_index_body = "\n".join(parameter_index_lines)
        body: list[str] = []
        team_body: list[str] = []
        dense_nodes = [node for node in self.graph.nodes if node.op in {"Dense", "DenseBiasActivation"}]
        none_accumulators = {node.id: 0 for node in dense_nodes}
        half_policies = [("infer_batch_half2", none_accumulators)]
        if emit_half2_heuristic:
            half_policies.append(("infer_batch_half2_heuristic", half2_accumulator_plan(self.graph)))
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
        local_mask_workspace_elements = self.sample_mask_plan.total_elements
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
        inline_mask_workspace_declaration = (
            f"    std::uint8_t mask_workspace[{local_mask_workspace_elements}];\n"
            if local_mask_workspace_elements > 0 else ""
        )
        batch_mask_workspace_declaration = (
            f"          std::uint8_t mask_workspace[{local_mask_workspace_elements}];\n"
            if local_mask_workspace_elements > 0 else ""
        )
        half_mask_workspace_declaration = (
            f"          ponni::TwoMask mask_workspace[{local_mask_workspace_elements}];\n"
            if local_mask_workspace_elements > 0 else ""
        )
        batch_launch = f"""    InputView const input_view = inputs;
    OutputView const output_view = outputs;
    ParameterView const parameters = parameters_;
    Kokkos::parallel_for(
        \"GeneratedModel::infer_batch\",
        Kokkos::RangePolicy<execution_space, Kokkos::LaunchBounds<MaxThreads, MinBlocks>>(0, batch_size),
        KOKKOS_LAMBDA(int linear) {{
          int const ibatch = linear % batch_size;
          int const iwork = linear / batch_size;
          (void) iwork;
          ponni::SArray<Scalar,num_inputs> inputs;
          ponni::SArray<Scalar,num_outputs> outputs;
          for (int i = 0; i < num_inputs; i++) inputs(i) = input_view(i,ibatch);
          ParameterView const parameters_ = parameters;
{batch_workspace_declaration}{batch_mask_workspace_declaration}{batch_body_text}
          for (int i = 0; i < num_outputs; i++) output_view(i,ibatch) = outputs(i);
        }});"""
        def make_half_launch(method_name: str) -> str:
            return f"""    InputView const input_view = inputs;
    OutputView const output_view = outputs;
    HalfParameterView const half_weights = half_parameters_;
    int const pair_count = (batch_size + 1) / 2;
    Kokkos::parallel_for(
        \"GeneratedModel::{method_name}\",
        Kokkos::RangePolicy<execution_space, Kokkos::LaunchBounds<MaxThreads, MinBlocks>>(0, pair_count),
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
{half_workspace_declaration}{half_mask_workspace_declaration}{half_body_texts[method_name]}
          for (int i = 0; i < num_outputs; i++) {{
            output_view(i,ibatch) = static_cast<Scalar>(outputs(i).low());
            if (has_high_lane) output_view(i,ibatch + 1) = static_cast<Scalar>(outputs(i).high());
          }}
        }});"""

        half_launches = {method_name: make_half_launch(method_name) for method_name, _ in half_policies}

        def make_half_method(method_name: str) -> str:
            return f"""  template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
  void {method_name}(InputView const & inputs, OutputView const & outputs) const {{
#if defined(KOKKOS_ENABLE_DEBUG)
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

namespace ponni::generated {{

template <class Scalar = float>
class {self.model_name} {{
public:
  static_assert(std::is_same_v<Scalar,float> || std::is_same_v<Scalar,double>,
                "GeneratedModel Scalar must be float or double");
  int static constexpr num_inputs = {num_inputs};
  int static constexpr num_outputs = {num_outputs};
  int static constexpr workspace_elements = {self.plan.total_elements};
  int static constexpr mask_workspace_elements = {self.mask_plan.total_elements};
  int static constexpr sample_local_workspace_elements = {local_workspace_elements};
  int static constexpr default_hierarchical_batch_tile = {self.default_batch_tile};
  int static constexpr maximum_hierarchical_batch_tile = {self.maximum_batch_tile};
  int static constexpr storage_parameter_elements = {payload_elements};
  int static constexpr learned_parameter_elements = {learned_offset};
  int static constexpr stored_scalar_code = {payload_scalar_code};
  int static constexpr stored_scalar_bytes = {4 if payload_scalar_code == 1 else 8};
  using scalar_type = Scalar;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using InputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using OutputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using ParameterView = Kokkos::View<Scalar*,Kokkos::LayoutRight,ponni::DeviceSpace>;
  using HalfParameterView =
      Kokkos::View<Kokkos::Experimental::half_t*,Kokkos::LayoutRight,ponni::DeviceSpace>;

private:
  ParameterView parameters_;
  HalfParameterView half_parameters_;

  KOKKOS_INLINE_FUNCTION
  static int parameter_storage_index(int index) {{
{parameter_index_body}
  }}

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

  KOKKOS_INLINE_FUNCTION static Scalar apply_round(Scalar value) {{
    if (!Kokkos::isfinite(value) || value == static_cast<Scalar>(0)) return value;
    Scalar const lower = Kokkos::floor(value);
    Scalar const fraction = value - lower;
    if (fraction < static_cast<Scalar>(0.5)) return lower;
    if (fraction > static_cast<Scalar>(0.5)) return lower + static_cast<Scalar>(1);
    Scalar const half_lower = lower * static_cast<Scalar>(0.5);
    return Kokkos::floor(half_lower) == half_lower ? lower : lower + static_cast<Scalar>(1);
  }}

  KOKKOS_INLINE_FUNCTION static Scalar apply_sign(Scalar value) {{
    if (Kokkos::isnan(value)) return value;
    if (value > static_cast<Scalar>(0)) return static_cast<Scalar>(1);
    if (value < static_cast<Scalar>(0)) return static_cast<Scalar>(-1);
    return static_cast<Scalar>(0);
  }}

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

  static constexpr int get_num_parameters() {{ return learned_parameter_elements; }}

  bool weights_loaded() const {{ return parameters_.is_allocated() && half_parameters_.is_allocated(); }}

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
    if (payload_bytes != static_cast<std::uint64_t>(storage_parameter_elements) * stored_scalar_bytes) {{
      return fail(\"weight-file element count does not match generated model\");
    }}
    Kokkos::View<Scalar*,Kokkos::LayoutRight,Kokkos::HostSpace>
        host_parameters(\"generated_parameters_host\", storage_parameter_elements);
    Kokkos::View<Kokkos::Experimental::half_t*,Kokkos::LayoutRight,Kokkos::HostSpace>
        host_half_parameters(\"generated_half_parameters_host\", storage_parameter_elements);
    for (int i = 0; i < storage_parameter_elements; i++) {{
      {'float' if payload_scalar_code == 1 else 'double'} stored_value;
      std::memcpy(&stored_value, payload + static_cast<std::size_t>(i) * stored_scalar_bytes, stored_scalar_bytes);
      host_parameters(i) = static_cast<Scalar>(stored_value);
      host_half_parameters(i) = Kokkos::Experimental::cast_to_half(static_cast<float>(stored_value));
    }}
    parameters_ = ParameterView(\"generated_parameters\", storage_parameter_elements);
    half_parameters_ = HalfParameterView(\"generated_half_parameters\", storage_parameter_elements);
    Kokkos::deep_copy(parameters_, host_parameters);
    Kokkos::deep_copy(half_parameters_, host_half_parameters);
    return true;
  }}

  bool save_parameters(std::string const & path, std::string * error = nullptr) const {{
    auto fail = [&](std::string const & message) {{
      if (error != nullptr) *error = message;
      return false;
    }};
    if (!weights_loaded()) return fail("generated model parameters are not loaded");
    auto const host_parameters = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), parameters_);
    int constexpr header_size = 32;
    std::uint64_t const payload_bytes =
        static_cast<std::uint64_t>(storage_parameter_elements) * stored_scalar_bytes;
    std::vector<unsigned char> bytes(header_size + static_cast<std::size_t>(payload_bytes));
    unsigned char const magic[8] = {{'P', 'N', 'N', 'W', 'G', 'T', '1', 0}};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    std::uint32_t const version = 1;
    std::uint32_t const scalar_code = stored_scalar_code;
    std::memcpy(bytes.data() + 8, &version, sizeof(version));
    std::memcpy(bytes.data() + 12, &scalar_code, sizeof(scalar_code));
    std::memcpy(bytes.data() + 16, &payload_bytes, sizeof(payload_bytes));
    for (int i = 0; i < storage_parameter_elements; i++) {{
      {'float' if payload_scalar_code == 1 else 'double'} const stored_value =
          static_cast<{'float' if payload_scalar_code == 1 else 'double'}>(host_parameters(i));
      std::memcpy(bytes.data() + header_size + static_cast<std::size_t>(i) * stored_scalar_bytes,
                  &stored_value, stored_scalar_bytes);
    }}
    std::uint64_t const payload_checksum = checksum(bytes.data() + header_size, payload_bytes);
    std::memcpy(bytes.data() + 24, &payload_checksum, sizeof(payload_checksum));
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return fail("cannot open parameter file for writing: " + path);
    stream.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) return fail("cannot write parameter file: " + path);
    return true;
  }}

  template <class ExecutionSpace = execution_space>
  void refresh_half_parameters(ExecutionSpace const & execution = ExecutionSpace()) {{
    static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                  "refresh_half_parameters requires a Kokkos execution space");
    static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,ponni::DeviceSpace>::accessible,
                  "refresh_half_parameters execution space cannot access the model parameter memory space");
    if (!parameters_.is_allocated() || !half_parameters_.is_allocated()) {{
      Kokkos::abort("GeneratedModel::refresh_half_parameters called before load_weights");
    }}
    ParameterView const parameters = parameters_;
    HalfParameterView const half_parameters = half_parameters_;
    Kokkos::parallel_for(
        "GeneratedModel::refresh_half_parameters",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, storage_parameter_elements),
        KOKKOS_LAMBDA(int i) {{
          half_parameters(i) = Kokkos::Experimental::cast_to_half(static_cast<float>(parameters(i)));
        }});
  }}

  template <class ParameterViewType>
  void get_parameters(ParameterViewType const & destination) const {{
    static_assert(Kokkos::is_view_v<ParameterViewType>, "get_parameters requires a Kokkos::View");
    static_assert(ParameterViewType::rank == 1, "get_parameters requires a rank-one Kokkos::View");
    using ParameterScalar = typename ParameterViewType::non_const_value_type;
    using MemorySpace = typename ParameterViewType::memory_space;
    using ExecutionSpace = typename MemorySpace::execution_space;
    static_assert(std::is_same_v<ParameterScalar,float> || std::is_same_v<ParameterScalar,double>,
                  "get_parameters destination scalar must be float or double");
    static_assert(!std::is_const_v<typename ParameterViewType::value_type>,
                  "get_parameters destination must be writable");
    static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,MemorySpace>::accessible,
                  "get_parameters destination execution space cannot access its memory space");
    if (!weights_loaded()) Kokkos::abort("GeneratedModel::get_parameters called before load_weights");
    if (destination.extent(0) != static_cast<std::size_t>(get_num_parameters())) {{
      Kokkos::abort("GeneratedModel::get_parameters destination extent is incorrect");
    }}
    Kokkos::View<Scalar*,ponni::DeviceSpace> learned_parameters(
        "generated_get_parameters_device", learned_parameter_elements);
    ParameterView const parameters = parameters_;
    execution_space const model_execution;
    Kokkos::parallel_for(
        "GeneratedModel::gather_parameters",
        Kokkos::RangePolicy<execution_space>(model_execution, 0, learned_parameter_elements),
        KOKKOS_LAMBDA(int i) {{ learned_parameters(i) = parameters(parameter_storage_index(i)); }});
    model_execution.fence("GeneratedModel::get_parameters gather");
    Kokkos::View<Scalar*,MemorySpace> converted_parameters(
        "generated_get_parameters_converted", learned_parameter_elements);
    Kokkos::deep_copy(converted_parameters, learned_parameters);
    ExecutionSpace const destination_execution;
    Kokkos::parallel_for(
        "GeneratedModel::get_parameters",
        Kokkos::RangePolicy<ExecutionSpace>(destination_execution, 0, learned_parameter_elements),
        KOKKOS_LAMBDA(int i) {{ destination(i) = static_cast<ParameterScalar>(converted_parameters(i)); }});
    destination_execution.fence("GeneratedModel::get_parameters conversion");
  }}

  template <class ParameterViewType,
            class ExecutionSpace = typename ParameterViewType::memory_space::execution_space>
  void set_parameters(ParameterViewType const & source,
                      ExecutionSpace const & execution = ExecutionSpace()) {{
    static_assert(Kokkos::is_view_v<ParameterViewType>, "set_parameters requires a Kokkos::View");
    static_assert(ParameterViewType::rank == 1, "set_parameters requires a rank-one Kokkos::View");
    using ParameterScalar = typename ParameterViewType::non_const_value_type;
    using MemorySpace = typename ParameterViewType::memory_space;
    static_assert(std::is_same_v<ParameterScalar,float> || std::is_same_v<ParameterScalar,double>,
                  "set_parameters source scalar must be float or double");
    static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                  "set_parameters requires a Kokkos execution space");
    static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,MemorySpace>::accessible,
                  "set_parameters execution space cannot access the source memory space");
    if (!weights_loaded()) Kokkos::abort("GeneratedModel::set_parameters called before load_weights");
    if (source.extent(0) != static_cast<std::size_t>(get_num_parameters())) {{
      Kokkos::abort("GeneratedModel::set_parameters source extent is incorrect");
    }}
    Kokkos::View<Scalar*,MemorySpace> converted_parameters(
        "generated_set_parameters_converted", learned_parameter_elements);
    Kokkos::parallel_for(
        "GeneratedModel::convert_set_parameters",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, learned_parameter_elements),
        KOKKOS_LAMBDA(int i) {{ converted_parameters(i) = static_cast<Scalar>(source(i)); }});
    execution.fence("GeneratedModel::set_parameters source conversion");
    Kokkos::View<Scalar*,ponni::DeviceSpace> learned_parameters(
        "generated_set_parameters_device", learned_parameter_elements);
    Kokkos::deep_copy(learned_parameters, converted_parameters);
    ParameterView const parameters = parameters_;
    execution_space const model_execution;
    Kokkos::parallel_for(
        "GeneratedModel::scatter_parameters",
        Kokkos::RangePolicy<execution_space>(model_execution, 0, learned_parameter_elements),
        KOKKOS_LAMBDA(int i) {{ parameters(parameter_storage_index(i)) = learned_parameters(i); }});
    refresh_half_parameters(model_execution);
    model_execution.fence("GeneratedModel::set_parameters update");
#if defined(KOKKOS_ENABLE_DEBUG)
    if (!parameters_are_finite()) Kokkos::abort("GeneratedModel::set_parameters received non-finite values");
#endif
  }}

  bool parameters_are_finite() const {{
    if (!weights_loaded()) return false;
    ParameterView const parameters = parameters_;
    int nonfinite_count = 0;
    Kokkos::parallel_reduce(
        "GeneratedModel::parameters_are_finite",
        Kokkos::RangePolicy<execution_space>(0, learned_parameter_elements),
        KOKKOS_LAMBDA(int i, int & count) {{
          if (!Kokkos::isfinite(parameters(parameter_storage_index(i)))) count++;
        }}, nonfinite_count);
    return nonfinite_count == 0;
  }}

  bool parameters_synchronized() const {{
    if (!weights_loaded()) return false;
    ParameterView const parameters = parameters_;
    HalfParameterView const half_parameters = half_parameters_;
    int mismatch_count = 0;
    Kokkos::parallel_reduce(
        "GeneratedModel::parameters_synchronized",
        Kokkos::RangePolicy<execution_space>(0, storage_parameter_elements),
        KOKKOS_LAMBDA(int i, int & count) {{
          auto const expected = Kokkos::Experimental::cast_to_half(static_cast<float>(parameters(i)));
          if (static_cast<float>(half_parameters(i)) != static_cast<float>(expected)) count++;
        }}, mismatch_count);
    return mismatch_count == 0;
  }}

  KOKKOS_INLINE_FUNCTION
  void infer_one(ponni::SArray<Scalar,num_inputs> const & inputs,
                 ponni::SArray<Scalar,num_outputs> & outputs) const {{
{inline_workspace_declaration}{inline_mask_workspace_declaration}{body_text}
  }}

  template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
  void infer_batch(InputView const & inputs, OutputView const & outputs) const {{
#if defined(KOKKOS_ENABLE_DEBUG)
    if (!weights_loaded()) Kokkos::abort(\"GeneratedModel::infer_batch called before load_weights\");
    if (inputs.extent(0) != num_inputs) Kokkos::abort(\"GeneratedModel input feature extent is incorrect\");
    if (outputs.extent(0) != num_outputs) Kokkos::abort(\"GeneratedModel output feature extent is incorrect\");
    if (inputs.extent(1) != outputs.extent(1)) Kokkos::abort(\"GeneratedModel batch extents differ\");
#endif
    int const batch_size = checked_batch_size(inputs);
{batch_launch}
  }}

{half_methods}

  template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
  void infer_batch_hierarchical(InputView const & inputs, OutputView const & outputs,
                                int batch_tile = default_hierarchical_batch_tile) const {{
#if defined(KOKKOS_ENABLE_DEBUG)
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
    using policy_type = Kokkos::TeamPolicy<
        execution_space, Kokkos::LaunchBounds<MaxThreads, MinBlocks>>;
    using member_type = typename policy_type::member_type;
    int const league_size = (batch_size + batch_tile - 1) / batch_tile;
    int const scalar_scratch_bytes = workspace_elements * batch_tile * static_cast<int>(sizeof(Scalar));
    int const mask_scratch_bytes = mask_workspace_elements * batch_tile * static_cast<int>(sizeof(std::uint8_t));
    int const scratch_bytes = scalar_scratch_bytes + mask_scratch_bytes;
    policy_type policy(league_size, Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    ParameterView const weights = parameters_;
    Kokkos::parallel_for(
        "GeneratedModel::infer_batch_hierarchical", policy,
        KOKKOS_LAMBDA(member_type const & team) {{
          int const batch_begin = team.league_rank() * batch_tile;
          int const remaining_batch = batch_size - batch_begin;
          int const active_batch = remaining_batch < batch_tile ? remaining_batch : batch_tile;
          Scalar * workspace = nullptr;
          std::uint8_t * mask_workspace = nullptr;
          if (scratch_bytes > 0) {{
            unsigned char * scratch = reinterpret_cast<unsigned char *>(team.team_shmem().get_shmem(scratch_bytes));
            workspace = reinterpret_cast<Scalar *>(scratch);
            mask_workspace = reinterpret_cast<std::uint8_t *>(scratch + scalar_scratch_bytes);
          }}
{team_body_text}
        }});
  }}
}};

}}  // namespace ponni::generated
"""
        output_path.write_text(text)


def emit_cpp(graph: Graph, plan: StoragePlan, sample_plan: StoragePlan,
             mask_plan: StoragePlan, sample_mask_plan: StoragePlan, schedule: DenseChainSchedule,
             offsets: dict[int, int], output_dir: Path, model_name: str,
             strategy: str, payload_elements: int, payload_scalar_code: int,
             default_batch_tile: int, maximum_batch_tile: int, streaming_output_threshold: int,
             explicit_half2_accumulators: dict[int, int] | None = None,
             emit_half2_heuristic: bool = False) -> Path:
    output_path = output_dir / f"{model_name}.hpp"
    CppEmitter(
        graph, plan, sample_plan, mask_plan, sample_mask_plan, schedule, offsets, model_name, strategy,
        default_batch_tile, maximum_batch_tile
    ).emit(
        output_path, payload_elements, payload_scalar_code, streaming_output_threshold,
        explicit_half2_accumulators, emit_half2_heuristic
    )
    return output_path


def emit_autotuner(output_dir: Path, model_name: str, payload_scalar_code: int,
                   maximum_batch_tile: int, has_explicit_half2: bool) -> Path:
    """Emit a standalone launch-bounds and hierarchical-tile throughput tuner."""
    identifier = _identifier(model_name)
    output_path = output_dir / f"{identifier}_autotune.cpp"
    scalar = "float" if payload_scalar_code == 1 else "double"
    launch_bounds = [(0, 0)] + [
        (maximum_threads, minimum_blocks)
        for maximum_threads in (64, 128, 256, 512, 1024)
        for minimum_blocks in (1, 2, 3, 4)
        if maximum_threads * minimum_blocks <= 1024
    ]
    batch_tiles: list[int] = []
    tile = 1
    while tile <= maximum_batch_tile:
        batch_tiles.append(tile)
        tile *= 2

    wrappers = """
template <unsigned MaxThreads, unsigned MinBlocks>
void run_batch(Model const & model, InputView const & inputs, OutputView const & outputs, int) {
  model.template infer_batch<MaxThreads, MinBlocks>(inputs, outputs);
}

template <unsigned MaxThreads, unsigned MinBlocks>
void run_half2(Model const & model, InputView const & inputs, OutputView const & outputs, int) {
  model.template infer_batch_half2<MaxThreads, MinBlocks>(inputs, outputs);
}

template <unsigned MaxThreads, unsigned MinBlocks>
void run_hierarchical(Model const & model, InputView const & inputs, OutputView const & outputs, int batch_tile) {
  model.template infer_batch_hierarchical<MaxThreads, MinBlocks>(inputs, outputs, batch_tile);
}
"""
    if has_explicit_half2:
        wrappers += """
template <unsigned MaxThreads, unsigned MinBlocks>
void run_half2_explicit(Model const & model, InputView const & inputs, OutputView const & outputs, int) {
  model.template infer_batch_half2_explicit<MaxThreads, MinBlocks>(inputs, outputs);
}
"""

    candidates: list[str] = []
    for maximum_threads, minimum_blocks in launch_bounds:
        candidates.append(
            f'    {{"infer_batch", {maximum_threads}, {minimum_blocks}, 0, '
            f'&run_batch<{maximum_threads}, {minimum_blocks}>}},'
        )
    for maximum_threads, minimum_blocks in launch_bounds:
        candidates.append(
            f'    {{"infer_batch_half2", {maximum_threads}, {minimum_blocks}, 0, '
            f'&run_half2<{maximum_threads}, {minimum_blocks}>}},'
        )
    if has_explicit_half2:
        for maximum_threads, minimum_blocks in launch_bounds:
            candidates.append(
                f'    {{"infer_batch_half2_explicit", {maximum_threads}, {minimum_blocks}, 0, '
                f'&run_half2_explicit<{maximum_threads}, {minimum_blocks}>}},'
            )
    for maximum_threads, minimum_blocks in launch_bounds:
        for batch_tile in batch_tiles:
            candidates.append(
                f'    {{"infer_batch_hierarchical", {maximum_threads}, {minimum_blocks}, {batch_tile}, '
                f'&run_hierarchical<{maximum_threads}, {minimum_blocks}>}},'
            )
    candidate_text = "\n".join(candidates)

    text = f"""// Generated deterministically by PONNI kokkos_nn. Edit candidate lists and run counts as desired.
#include \"{identifier}.hpp\"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {{

using Model = ponni::generated::{identifier}<{scalar}>;
using InputView = typename Model::InputView;
using OutputView = typename Model::OutputView;
using Runner = void (*)(Model const &, InputView const &, OutputView const &, int);

int constexpr default_batch_size = 1000000;
int constexpr warmup_runs = 3;
int constexpr timed_runs = 9;

struct Candidate {{
  char const * family;
  unsigned max_threads;
  unsigned min_blocks;
  int tile;
  Runner run;
}};

struct Result {{
  Candidate const * candidate;
  double median_ms;
}};

{wrappers}

std::vector<Candidate> candidates() {{
  return {{
{candidate_text}
  }};
}}

double time_candidate(Candidate const & candidate, Model const & model,
                      InputView const & inputs, OutputView const & outputs) {{
  for (int run = 0; run < warmup_runs; run++) candidate.run(model, inputs, outputs, candidate.tile);
  Kokkos::fence();

  std::vector<double> samples;
  samples.reserve(timed_runs);
  for (int run = 0; run < timed_runs; run++) {{
    auto const begin = std::chrono::steady_clock::now();
    candidate.run(model, inputs, outputs, candidate.tile);
    Kokkos::fence();
    auto const end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }}
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}}

void print_header() {{
  std::cout << std::left << std::setw(32) << "family"
            << std::right << std::setw(14) << "max_threads"
            << std::setw(13) << "min_blocks"
            << std::setw(8) << "tile"
            << std::setw(16) << "median_ms" << '\\n';
}}

void print_result(Result const & result) {{
  Candidate const & candidate = *result.candidate;
  std::cout << std::left << std::setw(32) << candidate.family
            << std::right << std::setw(14) << candidate.max_threads
            << std::setw(13) << candidate.min_blocks
            << std::setw(8) << candidate.tile
            << std::setw(16) << std::fixed << std::setprecision(6) << result.median_ms << '\\n';
}}

int parse_batch_size(int argc, char ** argv) {{
  if (argc < 2) return default_batch_size;
  long long const requested = std::stoll(argv[1]);
  if (requested < 1 || requested > std::numeric_limits<int>::max()) {{
    throw std::runtime_error("batch size must be in [1, INT_MAX]");
  }}
  return static_cast<int>(requested);
}}

}}  // namespace

int main(int argc, char ** argv) {{
  Kokkos::initialize(argc, argv);
  int status = 0;
  bool pool_initialized = false;
  {{
    try {{
      int const batch_size = parse_batch_size(argc, argv);
      std::string const weight_path = argc >= 3 ? argv[2] : "weights.bin";
      std::size_t const io_elements =
          static_cast<std::size_t>(Model::num_inputs + Model::num_outputs) * batch_size;
      std::size_t const parameter_bytes =
          static_cast<std::size_t>(Model::storage_parameter_elements) *
          (sizeof(typename Model::scalar_type) + sizeof(typename Model::HalfParameterView::non_const_value_type));
      std::size_t constexpr pool_margin_bytes = 1024 * 1024;
      ponni::init_device_pool(
          io_elements * sizeof(typename Model::scalar_type) + parameter_bytes + pool_margin_bytes);
      pool_initialized = true;
      Model model;
      std::string error;
      if (!model.load_weights(weight_path, &error)) {{
        throw std::runtime_error("failed to load " + weight_path + ": " + error);
      }}

      InputView inputs("autotune_inputs", Model::num_inputs, batch_size);
      OutputView outputs("autotune_outputs", Model::num_outputs, batch_size);
      auto host_inputs = Kokkos::create_mirror_view(inputs);
      std::mt19937 generator(20260803);
      std::uniform_real_distribution<{scalar}> distribution(static_cast<{scalar}>(-1), static_cast<{scalar}>(1));
      for (int i = 0; i < Model::num_inputs; i++) {{
        for (int ibatch = 0; ibatch < batch_size; ibatch++) host_inputs(i, ibatch) = distribution(generator);
      }}
      Kokkos::deep_copy(inputs, host_inputs);

      std::vector<Candidate> const configurations = candidates();
      std::vector<Result> results;
      results.reserve(configurations.size());
      for (Candidate const & candidate : configurations) {{
        results.push_back(Result{{&candidate, time_candidate(candidate, model, inputs, outputs)}});
      }}

      std::cout << "All results (batch_size=" << batch_size << ", timed_runs=" << timed_runs << ")\\n";
      print_header();
      for (Result const & result : results) print_result(result);

      std::map<std::string, Result const *> best;
      for (Result const & result : results) {{
        std::string const family(result.candidate->family);
        auto const found = best.find(family);
        if (found == best.end() || result.median_ms < found->second->median_ms) best[family] = &result;
      }}

      std::cout << "\\nBest result for each family\\n";
      print_header();
      for (auto const & entry : best) print_result(*entry.second);
    }} catch (std::exception const & exception) {{
      std::cerr << "autotuner: " << exception.what() << '\\n';
      status = 1;
    }}
  }}
  if (pool_initialized) ponni::finalize_device_pool();
  Kokkos::finalize();
  return status;
}}
"""
    output_path.write_text(text)
    return output_path
