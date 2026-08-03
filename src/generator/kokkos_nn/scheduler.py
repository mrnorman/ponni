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
    decisions: dict[int, ActivationDecision]
    pair_by_consumer: dict[int, int]
    eliminated_tensors: set[int]
    skipped_producers: set[int]
    recompute_extra_madds: int

    @property
    def has_streaming(self) -> bool:
        return bool(self.pair_by_consumer)

    def recompute_liveness_extensions(self, graph: Graph) -> dict[int, set[int]]:
        """Return delayed source uses introduced by branched recomputation.

        A sole-consumer streamed pair consumes its source before committing the
        destination accumulators, so the storage planner may reuse that source
        slot.  A recomputed producer, however, must read its source again for
        every branch and therefore extends the source lifetime through the last
        paired consumer.
        """
        nodes = {node.id: node for node in graph.nodes}
        extensions: dict[int, set[int]] = {}
        for decision in self.decisions.values():
            if decision.action != "recompute":
                continue
            producer = nodes[decision.producer_id]
            extensions.setdefault(producer.inputs[0], set()).update(decision.consumer_ids)
        return extensions

    def to_dict(self) -> dict[str, object]:
        counts = {action: 0 for action in ("materialize", "stream", "retain", "recompute")}
        for decision in self.decisions.values():
            counts[decision.action] += 1
        return {
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


def _dense_pair_eligible(graph: Graph, producer: Node, consumer: Node,
                         maximum_output_accumulators: int) -> bool:
    if producer.op != "DenseBiasActivation" or consumer.op not in {"Dense", "DenseBiasActivation"}:
        return False
    if producer.outputs[0] != consumer.inputs[0]:
        return False
    return graph.tensors[consumer.outputs[0]].sample_size <= maximum_output_accumulators


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


def schedule_dense_chains(graph: Graph, maximum_output_accumulators: int = 8,
                          recompute_madd_threshold: int = 64) -> DenseChainSchedule:
    """Choose deterministic materialize, stream, retain, and small recomputation actions."""
    graph.rebuild_links()
    nodes = {node.id: node for node in graph.nodes}
    selected_pairs: dict[int, int] = {}
    eliminated_tensors: set[int] = set()
    skipped_producers: set[int] = set()
    recomputed_tensors: dict[int, int] = {}
    blocked_nodes: set[int] = set()

    # Recompute only small branched dense activations whose consumers are all terminal dense nodes.
    # This keeps the rule explicit and prevents recursive or exponentially growing recomputation.
    for producer in graph.nodes:
        if producer.op != "DenseBiasActivation":
            continue
        tensor = graph.tensors[producer.outputs[0]]
        if len(tensor.consumers) < 2:
            continue
        consumers = [nodes[consumer_id] for consumer_id in tensor.consumers]
        if not all(_dense_pair_eligible(graph, producer, consumer, maximum_output_accumulators)
                   for consumer in consumers):
            continue
        terminal = all(
            not any(nodes[next_id].op in {"Dense", "DenseBiasActivation"}
                    for next_id in graph.tensors[consumer.outputs[0]].consumers)
            for consumer in consumers
        )
        producer_madds = (
            graph.tensors[producer.inputs[0]].sample_size * graph.tensors[producer.outputs[0]].sample_size
        )
        extra_madds = producer_madds * (len(consumers) - 1)
        if not terminal or extra_madds > recompute_madd_threshold:
            continue
        for consumer in consumers:
            selected_pairs[consumer.id] = producer.id
        eliminated_tensors.add(producer.outputs[0])
        skipped_producers.add(producer.id)
        recomputed_tensors[producer.outputs[0]] = extra_madds
        blocked_nodes.add(producer.id)
        blocked_nodes.update(consumer.id for consumer in consumers)

    # Remaining sole-consumer candidates form disjoint paths. Weighted matching avoids overlapping pairs and
    # chooses the schedule that eliminates the most activation elements on each path.
    candidates: dict[int, tuple[int, int]] = {}
    incoming: dict[int, int] = {}
    for producer in graph.nodes:
        if producer.id in blocked_nodes or producer.op != "DenseBiasActivation":
            continue
        tensor = graph.tensors[producer.outputs[0]]
        if len(tensor.consumers) != 1:
            continue
        consumer = nodes[tensor.consumers[0]]
        if consumer.id in blocked_nodes or not _dense_pair_eligible(
            graph, producer, consumer, maximum_output_accumulators
        ):
            continue
        candidates[producer.id] = (consumer.id, tensor.sample_size)
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
        if tensor_id in recomputed_tensors:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "recompute",
                "all consumers are terminal dense nodes and duplicated producer multiply-adds are within threshold",
                tensor.sample_size, recomputed_tensors[tensor_id],
            )
        elif tensor_id in eliminated_tensors:
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "stream",
                "selected by maximum-saved-activation matching for a legal dense pair",
                tensor.sample_size,
            )
        elif len(consumers) > 1:
            blockers: list[str] = []
            if producer.op != "DenseBiasActivation":
                blockers.append(f"producer {producer.op} is not an activated dense operation")
            else:
                consumer_nodes = [nodes[consumer_id] for consumer_id in consumers]
                if not all(_dense_pair_eligible(
                    graph, producer, consumer, maximum_output_accumulators
                ) for consumer in consumer_nodes):
                    blockers.append(
                        "not every consumer is an eligible dense operation within the output-accumulator limit"
                    )
                elif not all(
                    not any(nodes[next_id].op in {"Dense", "DenseBiasActivation"}
                            for next_id in graph.tensors[consumer.outputs[0]].consumers)
                    for consumer in consumer_nodes
                ):
                    blockers.append("not every dense consumer is terminal")
                producer_madds = graph.tensors[producer.inputs[0]].sample_size * tensor.sample_size
                extra_madds = producer_madds * (len(consumers) - 1)
                if extra_madds > recompute_madd_threshold:
                    blockers.append(
                        f"duplicated cost {extra_madds} multiply-adds exceeds threshold "
                        f"{recompute_madd_threshold}"
                    )
            decisions[tensor_id] = ActivationDecision(
                tensor_id, producer.id, consumers, "retain",
                f"{len(consumers)} consumers require retention; recomputation is not legal because "
                + "; ".join(blockers),
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
        sum(recomputed_tensors.values()),
    )
