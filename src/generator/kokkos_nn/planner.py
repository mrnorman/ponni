from __future__ import annotations

from dataclasses import dataclass

from .ir import Graph


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


def plan_storage(graph: Graph, excluded_tensors: set[int] | None = None,
                 additional_consumers: dict[int, set[int]] | None = None) -> StoragePlan:
    graph.rebuild_links()
    excluded_tensors = excluded_tensors or set()
    additional_consumers = additional_consumers or {}
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    intervals: list[tuple[int, int, int, int]] = []
    for tensor_id, tensor in graph.tensors.items():
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
