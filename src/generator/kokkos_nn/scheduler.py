"""Choose dense streaming and bounded recomputation before storage planning.

The scheduler trades extra arithmetic for smaller local arrays. Its output is
declarative: the emitter follows selected producer/consumer pairs, while the
planner uses eliminated tensors and extended liveness to size the remaining
arena accurately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import DType, Graph, Node
from .planner import plan_storage


DENSE_OPS = {"Dense", "DenseBiasActivation", "DenseEpilogue", "DenseResidualActivation"}


@dataclass(frozen=True)
class ActivationDecision:
    """Human-readable disposition of one materializable activation."""
    tensor_id: int
    producer_id: int
    consumer_ids: tuple[int, ...]
    action: str
    reason: str
    eliminated_elements: int = 0
    recompute_madds: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "producer_id": self.producer_id,
            "consumer_ids": list(self.consumer_ids),
            "action": self.action,
            "reason": self.reason,
            "eliminated_elements": self.eliminated_elements,
            "recompute_madds": self.recompute_madds,
        }


@dataclass
class DenseChainSchedule:
    """All streaming/recomputation choices needed by planning and emission."""
    aggressiveness: int
    decisions: dict[int, ActivationDecision]
    pair_by_consumer: dict[int, int]
    eliminated_tensors: set[int]
    skipped_producers: set[int]
    recompute_extra_madds: int

    def recompute_liveness_extensions(self, graph: Graph) -> dict[int, set[int]]:
        """Extend producer-input lifetimes through consumers that reevaluate them."""
        nodes = {node.id: node for node in graph.nodes}
        extensions: dict[int, set[int]] = {}
        for decision in self.decisions.values():
            if decision.action not in {"recompute", "stream"}:
                continue
            producer = nodes[decision.producer_id]
            for input_id in producer.inputs:
                if graph.tensors[input_id].is_constant:
                    continue
                extensions.setdefault(input_id, set()).update(decision.consumer_ids)
        return extensions

    def to_dict(self) -> dict[str, object]:
        counts = {action: 0 for action in ("materialize", "stream", "retain", "recompute")}
        for decision in self.decisions.values():
            counts[decision.action] += 1
        return {
            "workspace_reduction_aggressiveness": self.aggressiveness,
            "decision_counts": counts,
            "eliminated_tensors": sorted(self.eliminated_tensors),
            "eliminated_elements": sum(
                decision.eliminated_elements for decision in self.decisions.values()
            ),
            "recompute_extra_madds": self.recompute_extra_madds,
            "decisions": [
                self.decisions[tensor_id].to_dict() for tensor_id in sorted(self.decisions)
            ],
        }


def _dense_pair_eligible(graph: Graph, producer: Node, consumer: Node) -> bool:
    """Return whether a dense output can be accumulated directly by its consumer."""
    if producer.op not in DENSE_OPS or consumer.op not in DENSE_OPS:
        return False
    if producer.outputs[0] != consumer.inputs[0]:
        return False
    # Virtual Concat/Gather inputs are described by an index map rather than by
    # consumer.inputs[0].  The dense-pair emitter must not mistake the first
    # mapped source for the complete logical dense input.
    if "input_map" in producer.attributes or "input_map" in consumer.attributes:
        return False
    hidden_size = graph.tensors[producer.outputs[0]].sample_size
    output_size = graph.tensors[consumer.outputs[0]].sample_size
    return output_size < hidden_size


def _maximum_weight_path_edges(edges: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    """Select nonadjacent path edges, maximizing eliminated activation elements."""
    if not edges:
        return set()
    best: list[tuple[int, tuple[int, ...]]] = [(0, ()), (edges[0][2], (0,))]
    for index in range(1, len(edges)):
        without = best[index]
        base = best[index - 1] if index > 1 else (0, ())
        with_edge = (base[0] + edges[index][2], base[1] + (index,))
        best.append(with_edge if with_edge[0] > without[0] else without)
    return {(edges[index][0], edges[index][1]) for index in best[-1][1]}


def _terminal_dense_consumer(graph: Graph, nodes: dict[int, Node], consumer: Node) -> bool:
    return not any(
        nodes[next_id].op in DENSE_OPS
        for next_id in graph.tensors[consumer.outputs[0]].consumers
    )


def _recompute_candidates(graph: Graph, nodes: dict[int, Node], aggressiveness: int) -> list[Node]:
    """Find one-hop dense branches eligible for levels four and five."""
    candidates: list[Node] = []
    for producer in graph.nodes:
        if producer.op not in DENSE_OPS:
            continue
        tensor = graph.tensors[producer.outputs[0]]
        consumers = [nodes[consumer_id] for consumer_id in tensor.consumers]
        if len(consumers) < 2:
            continue
        if not all(
            consumer.op in DENSE_OPS and
            consumer.inputs[0] == tensor.id and "input_map" not in consumer.attributes
            for consumer in consumers
        ):
            continue
        if "input_map" in producer.attributes:
            continue
        if aggressiveness == 4 and (
            len(consumers) != 2 or
            not all(_terminal_dense_consumer(graph, nodes, consumer) for consumer in consumers)
        ):
            continue
        candidates.append(producer)
    return candidates


def _selected_linear_pairs(graph: Graph, nodes: dict[int, Node], aggressiveness: int,
                           blocked_nodes: set[int], base_excluded: set[int] | None = None,
                           extensions: dict[int, set[int]] | None = None) -> set[tuple[int, int]]:
    """Select non-overlapping linear dense pairs with minimum planned high-water."""
    candidates: dict[int, tuple[int, int]] = {}
    incoming: dict[int, int] = {}
    if aggressiveness < 2:
        return set()
    for producer in graph.nodes:
        if producer.id in blocked_nodes or producer.op not in DENSE_OPS:
            continue
        tensor = graph.tensors[producer.outputs[0]]
        if len(tensor.consumers) != 1:
            continue
        consumer = nodes[tensor.consumers[0]]
        if consumer.id in blocked_nodes or not _dense_pair_eligible(graph, producer, consumer):
            continue
        if aggressiveness == 2 and consumer.outputs[0] not in graph.outputs:
            continue
        output_size = graph.tensors[consumer.outputs[0]].sample_size
        candidates[producer.id] = (consumer.id, tensor.sample_size - output_size)
        incoming[consumer.id] = producer.id

    # Candidate edges form disjoint paths because each activation has exactly
    # one consumer. Split those paths before choosing nonadjacent edges.
    visited: set[int] = set()
    paths: list[list[tuple[int, int, int]]] = []
    starts = sorted(producer_id for producer_id in candidates if producer_id not in incoming)
    for start in starts:
        path: list[tuple[int, int, int]] = []
        producer_id = start
        while producer_id in candidates and producer_id not in visited:
            visited.add(producer_id)
            consumer_id, weight = candidates[producer_id]
            path.append((producer_id, consumer_id, weight))
            producer_id = consumer_id
        paths.append(path)
    edges = [edge for path in paths for edge in path]
    if len(edges) > 12:
        selected: set[tuple[int, int]] = set()
        for path in paths:
            selected.update(_maximum_weight_path_edges(path))
        return selected

    # For the small dense graphs PONNI targets, score every legal non-overlapping
    # subset by the arena high-water it actually produces.  This catches cases
    # where H-O is a poor proxy because unrelated live ranges dominate storage.
    best_key: tuple[int, int, tuple[tuple[int, int], ...]] | None = None
    best: set[tuple[int, int]] = set()
    for mask in range(1 << len(edges)):
        used_nodes: set[int] = set()
        selected_edges: set[tuple[int, int]] = set()
        eliminated = 0
        legal = True
        for index, (producer_id, consumer_id, weight) in enumerate(edges):
            if not mask & (1 << index):
                continue
            if producer_id in used_nodes or consumer_id in used_nodes:
                legal = False
                break
            used_nodes.update({producer_id, consumer_id})
            selected_edges.add((producer_id, consumer_id))
            eliminated += weight
        if not legal:
            continue
        excluded = set(base_excluded or ())
        excluded.update(nodes[producer_id].outputs[0] for producer_id, _ in selected_edges)
        candidate_extensions = {tensor_id: set(consumers) for tensor_id, consumers in (extensions or {}).items()}
        for producer_id, consumer_id in selected_edges:
            for input_id in nodes[producer_id].inputs:
                if graph.tensors[input_id].is_constant:
                    continue
                candidate_extensions.setdefault(input_id, set()).add(consumer_id)
        extent = plan_storage(
            graph, excluded, candidate_extensions, {DType.FLOAT32, DType.FLOAT64}, placement="heuristic",
        ).total_elements
        key = (extent, -eliminated, tuple(sorted(selected_edges)))
        if best_key is None or key < best_key:
            best_key = key
            best = selected_edges
    return best


def _scheduled_workspace_extent(graph: Graph, nodes: dict[int, Node], aggressiveness: int,
                                recomputed_producers: set[int]) -> int:
    excluded: set[int] = set()
    extensions: dict[int, set[int]] = {}
    blocked_nodes: set[int] = set()
    for producer_id in recomputed_producers:
        producer = nodes[producer_id]
        tensor = graph.tensors[producer.outputs[0]]
        excluded.add(tensor.id)
        for input_id in producer.inputs:
            if graph.tensors[input_id].is_constant:
                continue
            extensions.setdefault(input_id, set()).update(tensor.consumers)
        blocked_nodes.add(producer_id)
        blocked_nodes.update(tensor.consumers)
    selected_pairs = _selected_linear_pairs(
        graph, nodes, aggressiveness, blocked_nodes, excluded, extensions,
    )
    for producer_id, consumer_id in selected_pairs:
        excluded.add(nodes[producer_id].outputs[0])
        for input_id in nodes[producer_id].inputs:
            if graph.tensors[input_id].is_constant:
                continue
            extensions.setdefault(input_id, set()).add(consumer_id)
    return plan_storage(
        graph, excluded, extensions, {DType.FLOAT32, DType.FLOAT64},
    ).total_elements


def _select_recomputation(graph: Graph, nodes: dict[int, Node], aggressiveness: int) -> set[int]:
    """Select non-overlapping branches whose recomputation strictly saves space."""
    if aggressiveness < 4:
        return set()
    candidates = _recompute_candidates(graph, nodes, aggressiveness)
    if len(candidates) <= 8:
        # Small candidate sets are cheap enough to score exhaustively against
        # the real storage planner, with extra multiply-adds as the tie-breaker.
        best: set[int] = set()
        best_key = (_scheduled_workspace_extent(graph, nodes, aggressiveness, best), 0, ())
        for mask in range(1, 1 << len(candidates)):
            selected: set[int] = set()
            occupied: set[int] = set()
            extra_madds = 0
            legal = True
            for index, producer in enumerate(candidates):
                if not mask & (1 << index):
                    continue
                tensor = graph.tensors[producer.outputs[0]]
                candidate_nodes = {producer.id, *tensor.consumers}
                if occupied & candidate_nodes:
                    legal = False
                    break
                occupied.update(candidate_nodes)
                selected.add(producer.id)
                extra_madds += (
                    graph.tensors[producer.inputs[0]].sample_size * tensor.sample_size *
                    (len(tensor.consumers) - 1)
                )
            if not legal:
                continue
            key = (
                _scheduled_workspace_extent(graph, nodes, aggressiveness, selected),
                extra_madds,
                tuple(sorted(selected)),
            )
            if key < best_key:
                best_key = key
                best = selected
        return best

    # Larger graphs use a stable greedy fallback to keep compile time bounded.
    selected: set[int] = set()
    current = _scheduled_workspace_extent(graph, nodes, aggressiveness, selected)
    blocked_nodes: set[int] = set()
    for producer in candidates:
        tensor = graph.tensors[producer.outputs[0]]
        candidate_nodes = {producer.id, *tensor.consumers}
        if candidate_nodes & blocked_nodes:
            continue
        candidate_selected = selected | {producer.id}
        candidate = _scheduled_workspace_extent(
            graph, nodes, aggressiveness, candidate_selected,
        )
        if candidate >= current:
            continue
        selected = candidate_selected
        current = candidate
        blocked_nodes.update(candidate_nodes)
    return selected


def schedule_dense_chains(graph: Graph, workspace_reduction_aggressiveness: int = 3) -> DenseChainSchedule:
    """Select deterministic streaming and one-hop recomputation for levels one through five."""
    if workspace_reduction_aggressiveness not in range(1, 6):
        raise ValueError("workspace reduction aggressiveness must be an integer from 1 through 5")
    graph.rebuild_links()
    nodes = {node.id: node for node in graph.nodes}
    selected_pairs: dict[int, int] = {}
    eliminated_tensors: set[int] = set()
    skipped_producers: set[int] = set()
    blocked_nodes: set[int] = set()
    # Recomputed branches reserve their producer and all consumers before
    # linear streaming pairs are considered; the two mechanisms cannot overlap.
    recomputed_producers = _select_recomputation(graph, nodes, workspace_reduction_aggressiveness)
    recompute_madds: dict[int, int] = {}
    for producer_id in sorted(recomputed_producers):
        producer = nodes[producer_id]
        tensor = graph.tensors[producer.outputs[0]]
        for consumer_id in tensor.consumers:
            selected_pairs[consumer_id] = producer_id
        eliminated_tensors.add(tensor.id)
        skipped_producers.add(producer_id)
        blocked_nodes.add(producer_id)
        blocked_nodes.update(tensor.consumers)
        input_size = graph.tensors[producer.inputs[0]].sample_size
        recompute_madds[tensor.id] = input_size * tensor.sample_size * (len(tensor.consumers) - 1)

    base_excluded = {nodes[producer_id].outputs[0] for producer_id in recomputed_producers}
    extensions: dict[int, set[int]] = {}
    for producer_id in recomputed_producers:
        producer = nodes[producer_id]
        for input_id in producer.inputs:
            if graph.tensors[input_id].is_constant:
                continue
            extensions.setdefault(input_id, set()).update(graph.tensors[producer.outputs[0]].consumers)
    selected_linear_pairs = _selected_linear_pairs(
        graph, nodes, workspace_reduction_aggressiveness, blocked_nodes, base_excluded, extensions,
    )
    for producer_id, consumer_id in sorted(selected_linear_pairs):
        producer = nodes[producer_id]
        selected_pairs[consumer_id] = producer_id
        eliminated_tensors.add(producer.outputs[0])
        skipped_producers.add(producer_id)

    # Record every ordinary activation, not only optimized ones. Complete
    # decisions make the JSON report useful when diagnosing retained storage.
    decisions: dict[int, ActivationDecision] = {}
    for tensor_id, tensor in sorted(graph.tensors.items()):
        if tensor.is_input or tensor.is_constant or tensor_id in graph.outputs or tensor.producer is None:
            continue
        producer = nodes[tensor.producer]
        consumers = tuple(tensor.consumers)
        if tensor_id in recompute_madds:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "recompute",
                "one-hop recomputation strictly reduces planned workspace high-water",
                tensor.sample_size, recompute_madds[tensor_id],
            )
        elif tensor_id in eliminated_tensors:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "stream",
                "selected because dense-pair streaming reduces live scalar storage",
                tensor.sample_size,
            )
        elif len(consumers) > 1:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "retain",
                f"{len(consumers)} consumers require the activation to remain materialized",
            )
        else:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "materialize",
                "no selected legal streaming edge; materialization preserves dependencies",
            )

    return DenseChainSchedule(
        workspace_reduction_aggressiveness,
        decisions,
        selected_pairs,
        eliminated_tensors,
        skipped_producers,
        sum(recompute_madds.values()),
    )
