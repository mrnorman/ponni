from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ir import DType, Graph

if TYPE_CHECKING:
    from .scheduler import DenseChainSchedule


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

    def to_dict(self, scalar_bytes: int = 4) -> dict[str, object]:
        return {
            "total_elements": self.total_elements,
            "estimated_stack_bytes": self.total_elements * scalar_bytes,
            "estimated_team_scratch_bytes": self.total_elements * scalar_bytes,
            "external_workspace_bytes": 0,
            "reused_tensors": self.reused_tensors,
            "slots": {
                str(tensor_id): {"offset": slot.offset, "size": slot.size, "first_use": slot.first_use,
                                 "last_use": slot.last_use}
                for tensor_id, slot in sorted(self.slots.items())
            },
        }


@dataclass
class BatchTeamStoragePlan:
    team_size: int
    required_resident_teams: int
    scratch_budget_bytes: int
    local_plan: StoragePlan
    local_mask_plan: StoragePlan
    scratch_plan: StoragePlan
    scratch_mask_plan: StoragePlan
    scalar_bytes: int

    @property
    def local_bytes_per_sample(self) -> int:
        return self.local_plan.total_elements * self.scalar_bytes + self.local_mask_plan.total_elements

    @property
    def scratch_bytes_per_sample(self) -> int:
        return self.scratch_plan.total_elements * self.scalar_bytes + self.scratch_mask_plan.total_elements

    @property
    def scratch_bytes_per_team(self) -> int:
        return self.scratch_bytes_per_sample * self.team_size

    def to_dict(self) -> dict[str, object]:
        return {
            "team_size": self.team_size,
            "vector_length": 1,
            "launch_bounds": {"max_threads": self.team_size, "min_blocks": 0},
            "required_resident_teams_for_scratch_target": self.required_resident_teams,
            "scratch_budget_bytes_per_team": self.scratch_budget_bytes,
            "scratch_bytes_per_sample": self.scratch_bytes_per_sample,
            "scratch_bytes_per_team": self.scratch_bytes_per_team,
            "local_bytes_per_sample": self.local_bytes_per_sample,
            "local_plan": self.local_plan.to_dict(self.scalar_bytes),
            "local_mask_plan": self.local_mask_plan.to_dict(1),
            "scratch_plan": self.scratch_plan.to_dict(self.scalar_bytes),
            "scratch_mask_plan": self.scratch_mask_plan.to_dict(1),
        }


def plan_storage(graph: Graph, excluded_tensors: set[int] | None = None,
                 additional_consumers: dict[int, set[int]] | None = None,
                 dtypes: set[DType] | None = None) -> StoragePlan:
    graph.rebuild_links()
    excluded_tensors = excluded_tensors or set()
    additional_consumers = additional_consumers or {}
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    intervals: list[tuple[int, int, int, int]] = []
    for tensor_id, tensor in graph.tensors.items():
        if dtypes is not None and tensor.dtype not in dtypes:
            continue
        if tensor_id in excluded_tensors:
            continue
        if tensor.is_input or tensor.is_constant or tensor_id in graph.outputs or tensor.producer is None:
            continue
        first = position[tensor.producer]
        consumers = set(tensor.consumers) | additional_consumers.get(tensor_id, set())
        last = max((position[consumer] for consumer in consumers), default=first)
        intervals.append((first, last, tensor_id, tensor.sample_size))
    intervals.sort(key=lambda item: (item[0], item[2]))

    active: list[StorageSlot] = []
    free_blocks: list[tuple[int, int]] = []
    slots: dict[int, StorageSlot] = {}
    high_water = 0
    reused = 0

    def merge_free_blocks() -> None:
        nonlocal free_blocks
        merged: list[tuple[int, int]] = []
        for offset, size in sorted(free_blocks):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                old_offset, old_size = merged[-1]
                merged[-1] = (old_offset, old_size + size)
            else:
                merged.append((offset, size))
        free_blocks = merged

    for first, last, tensor_id, size in intervals:
        still_active: list[StorageSlot] = []
        for slot in active:
            if slot.last_use < first:
                free_blocks.append((slot.offset, slot.size))
            else:
                still_active.append(slot)
        active = still_active
        merge_free_blocks()

        offset = -1
        for block_index, (block_offset, block_size) in enumerate(free_blocks):
            if block_size < size:
                continue
            offset = block_offset
            del free_blocks[block_index]
            if block_size > size:
                free_blocks.append((block_offset + size, block_size - size))
            reused += 1
            break
        if offset < 0:
            # A free block at the top of the arena can grow in place. This is
            # particularly useful when alternating-width streamed dense pairs
            # leave a slightly smaller activation slot at the high-water mark.
            for block_index, (block_offset, block_size) in enumerate(free_blocks):
                if block_offset + block_size != high_water:
                    continue
                offset = block_offset
                high_water += size - block_size
                del free_blocks[block_index]
                reused += 1
                break
        if offset < 0:
            offset = high_water
            high_water += size
        slot = StorageSlot(tensor_id, offset, size, first, last)
        slots[tensor_id] = slot
        active.append(slot)

    return StoragePlan(slots, high_water, reused)


def plan_batch_team_storage(graph: Graph, schedule: DenseChainSchedule, scalar_bytes: int,
                            team_size: int, scratch_budget_bytes: int,
                            required_resident_teams: int, beam_width: int = 16) -> BatchTeamStoragePlan:
    """Choose a deterministic mixed local/team-scratch activation plan.

    Each tensor is assigned wholly to one arena. The bounded beam search keeps
    storage planning inexpensive while considering combinations that a simple
    one-tensor-at-a-time greedy allocator can miss due to arena fragmentation.
    """
    floating = {DType.FLOAT32, DType.FLOAT64}
    liveness_extensions = schedule.recompute_liveness_extensions(graph)
    base_float = plan_storage(
        graph, schedule.eliminated_tensors, liveness_extensions, floating,
    )
    base_mask = plan_storage(graph, dtypes={DType.BOOL})
    items = tuple(
        [(tensor_id, False) for tensor_id in sorted(base_float.slots)] +
        [(tensor_id, True) for tensor_id in sorted(base_mask.slots)]
    )
    all_float = set(base_float.slots)
    all_mask = set(base_mask.slots)
    cache: dict[frozenset[tuple[int, bool]], BatchTeamStoragePlan] = {}

    def evaluate(selected: frozenset[tuple[int, bool]]) -> BatchTeamStoragePlan:
        if selected in cache:
            return cache[selected]
        scratch_float = {tensor_id for tensor_id, is_mask in selected if not is_mask}
        scratch_mask = {tensor_id for tensor_id, is_mask in selected if is_mask}
        local_plan = plan_storage(
            graph,
            schedule.eliminated_tensors | scratch_float,
            liveness_extensions,
            floating,
        )
        local_mask_plan = plan_storage(graph, scratch_mask, dtypes={DType.BOOL})
        scratch_plan = plan_storage(
            graph,
            schedule.eliminated_tensors | (all_float - scratch_float),
            liveness_extensions,
            floating,
        )
        scratch_mask_plan = plan_storage(graph, all_mask - scratch_mask, dtypes={DType.BOOL})
        plan = BatchTeamStoragePlan(
            team_size, required_resident_teams, scratch_budget_bytes,
            local_plan, local_mask_plan, scratch_plan, scratch_mask_plan, scalar_bytes,
        )
        cache[selected] = plan
        return plan

    states = {frozenset()}
    best = evaluate(frozenset())
    for _ in range(len(items)):
        expanded = set(states)
        for selected in states:
            for item in items:
                if item not in selected:
                    expanded.add(selected | {item})
        feasible = [selected for selected in expanded if evaluate(selected).scratch_bytes_per_team <= scratch_budget_bytes]
        if not feasible:
            break
        feasible.sort(key=lambda selected: (
            evaluate(selected).local_bytes_per_sample,
            evaluate(selected).scratch_bytes_per_team,
            len(selected),
            tuple(sorted(selected)),
        ))
        candidate = evaluate(feasible[0])
        if (
            candidate.local_bytes_per_sample,
            candidate.scratch_bytes_per_team,
        ) < (
            best.local_bytes_per_sample,
            best.scratch_bytes_per_team,
        ):
            best = candidate

        # Keep intermediate combinations even when they have not reduced the
        # local high-water mark yet: two such moves can free an overlapping
        # arena that neither move frees independently.
        states = set(feasible[:beam_width])
        if not states:
            break
    return best
