from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

from .ir import DType, Graph


@dataclass
class StorageSlot:
    tensor_id: int
    offset: int
    size: int
    first_use: int
    last_use: int


@dataclass
class StoragePlan:
    slots: dict[int, StorageSlot]
    total_elements: int
    reused_tensors: int
    placement_strategy: str = "heuristic"
    heuristic_elements: int | None = None
    optimality_proven: bool = False

    def to_dict(self, scalar_bytes: int = 4) -> dict[str, object]:
        return {
            "total_elements": self.total_elements,
            "estimated_stack_bytes": self.total_elements * scalar_bytes,
            "external_workspace_bytes": 0,
            "reused_tensors": self.reused_tensors,
            "placement_strategy": self.placement_strategy,
            "heuristic_elements": self.heuristic_elements,
            "optimality_proven": self.optimality_proven,
            "slots": {
                str(tensor_id): {"offset": slot.offset, "size": slot.size, "first_use": slot.first_use,
                                 "last_use": slot.last_use}
                for tensor_id, slot in sorted(self.slots.items())
            },
        }


def plan_storage(graph: Graph, excluded_tensors: set[int] | None = None,
                 additional_consumers: dict[int, set[int]] | None = None,
                 dtypes: set[DType] | None = None,
                 placement: Literal["native", "heuristic", "exact"] = "native",
                 exact_group_limit: int = 9) -> StoragePlan:
    graph.rebuild_links()
    excluded_tensors = excluded_tensors or set()
    additional_consumers = additional_consumers or {}
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    eligible = {
        tensor_id for tensor_id, tensor in graph.tensors.items()
        if (dtypes is None or tensor.dtype in dtypes) and tensor_id not in excluded_tensors and
        not tensor.is_input and not tensor.is_constant and tensor_id not in graph.outputs and
        tensor.producer is not None
    }
    parent = {tensor_id: tensor_id for tensor_id in eligible}

    def find(tensor_id: int) -> int:
        while parent[tensor_id] != tensor_id:
            parent[tensor_id] = parent[parent[tensor_id]]
            tensor_id = parent[tensor_id]
        return tensor_id

    def join(output_id: int, input_id: int) -> None:
        output_root = find(output_id)
        input_root = find(input_id)
        if output_root != input_root:
            parent[output_root] = input_root

    pointwise = {
        "Abs", "Acos", "Acosh", "Add", "And", "Asin", "Asinh", "Atan", "Atanh", "BatchNormalization",
        "Cast", "Ceil", "Celu", "Clip", "CompareSelect", "Cos", "Cosh", "Div", "ElementwiseChain", "Elu",
        "Equal", "Erf", "Exp", "Floor", "Gelu", "Greater", "GreaterOrEqual", "HardSigmoid", "HardSwish",
        "IsInf", "IsNaN", "LeakyRelu", "Less", "LessOrEqual", "Log", "Max", "Mean", "Min", "Mish",
        "Mul", "Neg", "Not", "Or", "PRelu", "PointwiseRegion", "Pow", "Reciprocal", "Relu",
        "ResidualAddActivation", "Round",
        "Selu", "Sigmoid", "Sign", "Silu", "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Sub", "Sum",
        "Tan", "Tanh", "ThresholdedRelu", "Where", "Xor",
    }
    whole_vector_in_place = {"LayerNormalization", "LogSoftmax", "LpNormalization", "Softmax"}
    for node in graph.nodes:
        output_id = node.outputs[0]
        if output_id not in eligible:
            continue
        if node.op == "DenseResidualActivation":
            candidates = [int(node.attributes["residual"])]
            if candidates[0] == node.inputs[0]:
                continue
        elif node.op in pointwise | whole_vector_in_place:
            candidates = list(node.inputs)
        else:
            continue
        output = graph.tensors[output_id]
        for input_id in candidates:
            if input_id not in eligible:
                continue
            value = graph.tensors[input_id]
            consumers = set(value.consumers) | additional_consumers.get(input_id, set())
            last_consumer = max(consumers, key=lambda consumer: position[consumer]) if consumers else None
            if (last_consumer != node.id or value.dtype != output.dtype or
                    value.sample_size < output.sample_size):
                continue
            join(output_id, input_id)
            break

    groups: dict[int, list[int]] = {}
    for tensor_id in eligible:
        groups.setdefault(find(tensor_id), []).append(tensor_id)
    intervals: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    for root, members in groups.items():
        first = min(position[graph.tensors[tensor_id].producer] for tensor_id in members)
        consumers: set[int] = set()
        for tensor_id in members:
            consumers.update(graph.tensors[tensor_id].consumers)
            consumers.update(additional_consumers.get(tensor_id, set()))
        last = max((position[consumer] for consumer in consumers), default=first)
        size = max(graph.tensors[tensor_id].sample_size for tensor_id in members)
        intervals.append((first, last, root, size, tuple(sorted(members))))
    intervals.sort(key=lambda item: (item[0], item[2]))

    def place(order) -> tuple[int, dict[int, int]]:
        placed: list[tuple[int, int, int, int]] = []
        offsets: dict[int, int] = {}
        high_water = 0
        for first, last, root, size, _ in order:
            overlapping = [item for item in placed if not (last < item[0] or item[1] < first)]
            candidates = sorted({0, *(offset + extent for _, _, offset, extent in overlapping)})
            offset = next(
                candidate for candidate in candidates
                if all(candidate + size <= other_offset or other_offset + other_size <= candidate
                       for _, _, other_offset, other_size in overlapping)
            )
            offsets[root] = offset
            placed.append((first, last, offset, size))
            high_water = max(high_water, offset + size)
        return high_water, offsets

    orders = [
        intervals,
        sorted(intervals, key=lambda item: (-item[3], item[0], item[2])),
        sorted(intervals, key=lambda item: (-(item[1] - item[0] + 1) * item[3], -item[3], item[2])),
    ]
    heuristic_elements, heuristic_offsets = min(
        (place(order) for order in orders), key=lambda result: result[0]
    )
    total_elements, offsets = heuristic_elements, heuristic_offsets
    strategy = "heuristic"
    optimality_proven = len(intervals) <= 1

    # Arena placement is a small dynamic-storage-allocation problem. Exhaustive
    # bottom-left placement is cheap for the small DAGs PONNI targets and closes
    # real fragmentation gaps left by any fixed list of greedy orderings.
    enumeration_limit = 7 if placement == "native" else exact_group_limit
    if placement != "heuristic" and len(intervals) <= enumeration_limit:
        total_elements, offsets = min(
            (place(order) for order in permutations(intervals)),
            key=lambda result: (result[0], tuple(sorted(result[1].items()))),
            default=(0, {}),
        )
        strategy = "exact-enumeration"
        optimality_proven = True
    elif placement == "exact":
        cp_sat_result = _cp_sat_place(intervals)
        if cp_sat_result is not None:
            candidate_elements, candidate_offsets, optimality_proven = cp_sat_result
            if candidate_elements <= heuristic_elements:
                total_elements, offsets = candidate_elements, candidate_offsets
                strategy = "cp-sat" if optimality_proven else "cp-sat-feasible"
            else:
                optimality_proven = False
                strategy = "heuristic-cp-sat-timeout"
        else:
            strategy = "heuristic-exact-limit"
    slots: dict[int, StorageSlot] = {}
    for first, last, root, size, members in intervals:
        for tensor_id in members:
            slots[tensor_id] = StorageSlot(tensor_id, offsets[root], graph.tensors[tensor_id].sample_size, first, last)
    reused = len(slots) - len(set(slot.offset for slot in slots.values()))
    return StoragePlan(
        slots, total_elements, reused, strategy, heuristic_elements, optimality_proven,
    )


def _cp_sat_place(intervals) -> tuple[int, dict[int, int], bool] | None:
    """Run optional OR-Tools in isolation from framework-native libraries."""
    # TensorFlow and OR-Tools can leave incompatible native runtime state in the
    # same process. A fresh interpreter also contains failures in this optional
    # backend so native placement can retain its heuristic fallback.
    worker = (
        "import json, sys; "
        "from kokkos_nn.planner import _cp_sat_place_in_process; "
        "json.dump(_cp_sat_place_in_process(json.load(sys.stdin)), sys.stdout)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", worker],
            input=json.dumps(intervals),
            text=True,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if result is None:
        return None
    elements, offsets, proven = result
    return int(elements), {int(root): int(offset) for root, offset in offsets.items()}, bool(proven)


def _cp_sat_place_in_process(intervals) -> tuple[int, dict[int, int], bool] | None:
    """Use optional OR-Tools in a clean process without a hard dependency."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    if not intervals:
        return 0, {}, True
    upper_bound = sum(item[3] for item in intervals)
    model = cp_model.CpModel()
    offsets = {
        root: model.new_int_var(0, upper_bound - size, f"offset_{root}")
        for _, _, root, size, _ in intervals
    }
    high_water = model.new_int_var(0, upper_bound, "high_water")
    for _, _, root, size, _ in intervals:
        model.add(high_water >= offsets[root] + size)
    # Every simultaneously live set must fit in the arena. This weighted-clique
    # lower bound is exact for fully overlapping layouts and substantially
    # shortens proofs for the general interval case.
    event_positions = sorted({item[0] for item in intervals} | {item[1] for item in intervals})
    concurrent_lower_bound = max(
        sum(size for first, last, _, size, _ in intervals if first <= position <= last)
        for position in event_positions
    )
    model.add(high_water >= concurrent_lower_bound)
    for index, (first, last, root, size, _) in enumerate(intervals):
        for other_first, other_last, other_root, other_size, _ in intervals[index + 1:]:
            if last < other_first or other_last < first:
                continue
            before = model.new_bool_var(f"before_{root}_{other_root}")
            model.add(offsets[root] + size <= offsets[other_root]).only_enforce_if(before)
            model.add(offsets[other_root] + other_size <= offsets[root]).only_enforce_if(before.Not())
    model.minimize(high_water)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return (
        int(solver.value(high_water)),
        {root: int(solver.value(offset)) for root, offset in offsets.items()},
        status == cp_model.OPTIMAL,
    )
