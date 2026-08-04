from __future__ import annotations

from dataclasses import dataclass

from .ir import Graph, Node


@dataclass(frozen=True)
class ActivationDecision:
    tensor_id: int
    producer_id: int
    consumer_ids: tuple[int, ...]
    action: str
    reason: str
    eliminated_elements: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "producer_id": self.producer_id,
            "consumer_ids": list(self.consumer_ids),
            "action": self.action,
            "reason": self.reason,
            "eliminated_elements": self.eliminated_elements,
        }


@dataclass
class DenseChainSchedule:
    decisions: dict[int, ActivationDecision]
    pair_by_consumer: dict[int, int]
    eliminated_tensors: set[int]
    skipped_producers: set[int]

    @property
    def has_streaming(self) -> bool:
        return bool(self.pair_by_consumer)

    def to_dict(self) -> dict[str, object]:
        counts = {action: 0 for action in ("materialize", "stream", "retain")}
        for decision in self.decisions.values():
            counts[decision.action] += 1
        return {
            "decision_counts": counts,
            "eliminated_tensors": sorted(self.eliminated_tensors),
            "eliminated_elements": sum(
                decision.eliminated_elements for decision in self.decisions.values()
            ),
            "decisions": [
                self.decisions[tensor_id].to_dict() for tensor_id in sorted(self.decisions)
            ],
        }


def _dense_pair_eligible(graph: Graph, producer: Node, consumer: Node) -> bool:
    if producer.op != "DenseBiasActivation" or consumer.op not in {"Dense", "DenseBiasActivation"}:
        return False
    if producer.outputs[0] != consumer.inputs[0]:
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


def schedule_dense_chains(graph: Graph) -> DenseChainSchedule:
    """Stream non-overlapping dense pairs only when doing so reduces live scalar storage."""
    graph.rebuild_links()
    nodes = {node.id: node for node in graph.nodes}
    selected_pairs: dict[int, int] = {}
    eliminated_tensors: set[int] = set()
    skipped_producers: set[int] = set()
    # Sole-consumer candidates form disjoint paths. Weighted matching chooses
    # non-overlapping pairs with the largest net reduction in live scalars.
    candidates: dict[int, tuple[int, int]] = {}
    incoming: dict[int, int] = {}
    for producer in graph.nodes:
        if producer.op != "DenseBiasActivation":
            continue
        tensor = graph.tensors[producer.outputs[0]]
        if len(tensor.consumers) != 1:
            continue
        consumer = nodes[tensor.consumers[0]]
        if not _dense_pair_eligible(graph, producer, consumer):
            continue
        output_size = graph.tensors[consumer.outputs[0]].sample_size
        candidates[producer.id] = (consumer.id, tensor.sample_size - output_size)
        incoming[consumer.id] = producer.id

    visited: set[int] = set()
    selected_linear_pairs: set[tuple[int, int]] = set()
    starts = sorted(producer_id for producer_id in candidates if producer_id not in incoming)
    for start in starts:
        path: list[tuple[int, int, int]] = []
        producer_id = start
        while producer_id in candidates and producer_id not in visited:
            visited.add(producer_id)
            consumer_id, weight = candidates[producer_id]
            path.append((producer_id, consumer_id, weight))
            producer_id = consumer_id
        selected_linear_pairs.update(_maximum_weight_path_edges(path))

    for producer_id, consumer_id in sorted(selected_linear_pairs):
        producer = nodes[producer_id]
        selected_pairs[consumer_id] = producer_id
        eliminated_tensors.add(producer.outputs[0])
        skipped_producers.add(producer_id)

    decisions: dict[int, ActivationDecision] = {}
    for tensor_id, tensor in sorted(graph.tensors.items()):
        if tensor.is_input or tensor.is_constant or tensor_id in graph.outputs or tensor.producer is None:
            continue
        producer = nodes[tensor.producer]
        consumers = tuple(tensor.consumers)
        if tensor_id in eliminated_tensors:
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
        decisions,
        selected_pairs,
        eliminated_tensors,
        skipped_producers,
    )
