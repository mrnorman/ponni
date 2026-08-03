from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .emitter import (
    emit_cpp,
    estimate_tensorcore_scratch_bytes,
    find_tensorcore_dense_chain,
    half2_accumulator_plan,
)
from .errors import CompilerError
from .importer import import_onnx
from .interpreter import run_graph
from .passes import optimize
from .planner import plan_storage
from .scheduler import DenseChainSchedule, schedule_dense_chains
from .weights import write_weights


def load_and_optimize(model_path: str | Path, disabled_passes: set[str] | None = None):
    original = import_onnx(model_path)
    optimized, pass_report = optimize(original, disabled_passes)
    return original, optimized, pass_report


def _shape_string(shape: tuple[object, ...]) -> str:
    return "(" + ", ".join(getattr(dim, "name", str(dim)) for dim in shape) + ")"


def _fusion_rejections(graph, schedule: DenseChainSchedule) -> list[str]:
    graph.rebuild_links()
    reasons: list[str] = []
    for node in graph.nodes:
        if node.op not in {"Dense", "DenseBiasActivation", "Add"}:
            continue
        consumers = graph.tensors[node.outputs[0]].consumers
        if len(consumers) > 1:
            decision = schedule.decisions.get(node.outputs[0])
            if decision is not None and decision.action != "retain":
                continue
            reasons.append(
                f"node {node.id} ({node.op}) retains its output because it has {len(consumers)} consumers"
                + (f": {decision.reason}" if decision is not None else "")
            )
    return reasons


def _report(original, optimized, pass_report, strategy: str, plan, sample_plan,
            schedule: DenseChainSchedule, scalar_bytes: int,
            default_batch_tile: int = 1, maximum_batch_tile: int = 1,
            streaming_output_threshold: int = 8) -> dict[str, Any]:
    input_tensor = original.tensors[original.inputs[0]]
    output_tensor = original.tensors[original.outputs[0]]
    parameter_count = sum(constant.values.size for constant in optimized.constants.values())
    canonical_counts = dict(sorted(Counter(node.op for node in optimized.nodes).items()))
    fused_ops = sum(
        count for op, count in canonical_counts.items()
        if op in {"DenseBiasActivation", "ElementwiseChain", "ResidualAddActivation"}
    )
    scheduled_consumers = [optimized.node_by_id(node_id) for node_id in schedule.pair_by_consumer]
    streaming_pair = len(optimized.nodes) == 2 and len(scheduled_consumers) == 1
    streaming_tail = bool(
        scheduled_consumers and scheduled_consumers[-1].id == optimized.nodes[-1].id and
        schedule.pair_by_consumer[scheduled_consumers[-1].id] == optimized.nodes[-2].id
    )
    return {
        "model_inputs": [{"name": input_tensor.name, "shape": _shape_string(input_tensor.shape),
                          "dtype": input_tensor.dtype.value}],
        "model_outputs": [{"name": output_tensor.name, "shape": _shape_string(output_tensor.shape),
                            "dtype": output_tensor.dtype.value}],
        "operator_counts": original.metadata.get("operator_counts", {}),
        "parameter_count": int(parameter_count),
        "original_onnx_node_count": int(original.metadata.get("original_node_count", len(original.nodes))),
        "canonical_node_count": len(original.nodes),
        "fused_node_count": len(optimized.nodes),
        "fused_operation_count": fused_ops,
        "canonical_operations": [node.op for node in original.nodes],
        "optimized_operations": [node.op for node in optimized.nodes],
        "passes": pass_report,
        "storage": plan.to_dict(scalar_bytes),
        "dense_chain_schedule": schedule.to_dict(),
        "sample_local_storage": {
            "workspace_elements": sample_plan.total_elements,
            "workspace_bytes": sample_plan.total_elements * scalar_bytes,
            "plan": sample_plan.to_dict(scalar_bytes),
            "batch_input_staging_elements": optimized.tensors[optimized.inputs[0]].sample_size,
            "batch_input_staging_bytes": optimized.tensors[optimized.inputs[0]].sample_size * scalar_bytes,
            "streaming_dense_pair": streaming_pair,
            "streaming_dense_tail": streaming_tail,
            "streamed_dense_pairs": len(schedule.pair_by_consumer),
            "recomputed_activations": sum(
                decision.action == "recompute" for decision in schedule.decisions.values()
            ),
            "streaming_output_accumulators": max(
                (optimized.tensors[node.outputs[0]].sample_size for node in scheduled_consumers),
                default=0,
            ),
        },
        "recommended_batched_target": {
            "sample-local": "infer_batch",
            "team": "infer_batch_hierarchical",
            "tensorcore": "infer_batch_tensorcore",
            "half2": "infer_batch_half2_heuristic",
        }.get(strategy, strategy),
        "generated_targets": [
            "infer_one", "infer_batch", "infer_batch_hierarchical", "infer_batch_tensorcore",
            "infer_batch_half2", "infer_batch_half2_heuristic"
        ],
        "execution_strategies": {
            "infer_one": "device-inline fixed SArray inference with local planned storage",
            "infer_batch": (
                "View-based sample-local inference per RangePolicy batch iteration, with one fixed SArray input "
                "staging pass and streaming dense emission when legal"
            ),
            "infer_batch_hierarchical": (
                "one TeamPolicy team per batch tile, TeamThreadRange over neuron-by-batch work with batch "
                "fastest, and batch-strided planned per-team scratch"
            ),
            "infer_batch_tensorcore": (
                "explicit raw CUDA Ampere WMMA TF32 kernel over 16-sample batch tiles; available for legal "
                "float32 two- or three-dense chains and selected only by explicit request"
            ),
            "infer_batch_half2": (
                "Kokkos RangePolicy over pairs of adjacent batch samples using native CUDA/HIP half2 packed "
                "FP16 multiply-accumulate with one dependent accumulation chain per dense dot product"
            ),
            "infer_batch_half2_heuristic": (
                "Kokkos half2 inference with a generated per-dense accumulator count selected from measured "
                "dot-length thresholds"
            ),
        },
        "hierarchical_batch_tiling": {
            "default_tile": default_batch_tile,
            "maximum_tile": maximum_batch_tile,
            "scratch_bytes_at_default": plan.total_elements * scalar_bytes * default_batch_tile,
            "index_order": "linear = neuron * active_batch + local_batch",
        },
        "random_access_weights": (
            "not selected: current dense loops use short, regular output-major weight traversals without a "
            "demonstrated benefit from Kokkos::RandomAccess"
        ),
        "batch_fastest": True,
        "fusion_rejections": _fusion_rejections(optimized, schedule),
        "rejected_constructs": [],
    }


def validate_model(model_path: str | Path, disabled_passes: set[str] | None = None) -> dict[str, Any]:
    original, optimized, pass_report = load_and_optimize(model_path, disabled_passes)
    plan = plan_storage(optimized)
    schedule = schedule_dense_chains(optimized)
    sample_plan = plan_storage(
        optimized, schedule.eliminated_tensors,
        schedule.recompute_liveness_extensions(optimized),
    )
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype.value == "float32" else 8
    return _report(original, optimized, pass_report, "not-selected", plan, sample_plan, schedule, scalar_bytes)


def _power_of_two_at_most(value: int) -> int:
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _hierarchical_batch_tiles(_maximum_parallel_neurons: int, stack_bytes: int,
                              max_team_scratch_bytes: int) -> tuple[int, int]:
    scratch_limited = 32 if stack_bytes == 0 else max_team_scratch_bytes // stack_bytes
    maximum_tile = _power_of_two_at_most(max(1, min(32, scratch_limited)))
    # Ampere measurements of I -> I -> I -> 3 networks for I=4--128 and batches
    # 10^4--10^6 favor 32, except for a small width-32 advantage at 16. Keep the
    # more robust 32 default; the device scratch limit remains authoritative.
    measured_tile = 32
    return min(maximum_tile, measured_tile), maximum_tile


_HALF2_ACCUMULATOR_CHOICES = {0, 2, 4, 8, 16, 32}


def _explicit_half2_accumulator_map(
    graph, specification: int | str | list[int] | tuple[int, ...] | None
) -> dict[int, int] | None:
    if specification is None:
        return None
    dense_nodes = [node for node in graph.nodes if node.op in {"Dense", "DenseBiasActivation"}]
    if isinstance(specification, int):
        values = [specification]
    elif isinstance(specification, str):
        try:
            values = [int(value.strip()) for value in specification.split(",") if value.strip()]
        except ValueError as exc:
            raise CompilerError(
                "--half2-accumulators must be one integer or a comma-separated list of integers"
            ) from exc
    else:
        values = [int(value) for value in specification]
    if not values:
        raise CompilerError("--half2-accumulators cannot be empty")
    invalid = [value for value in values if value not in _HALF2_ACCUMULATOR_CHOICES]
    if invalid:
        choices = ", ".join(str(value) for value in sorted(_HALF2_ACCUMULATOR_CHOICES))
        raise CompilerError(
            f"unsupported half2 accumulator count {invalid[0]}; supported counts are {choices}"
        )
    if len(values) == 1:
        values *= len(dense_nodes)
    elif len(values) != len(dense_nodes):
        raise CompilerError(
            f"--half2-accumulators provided {len(values)} counts for {len(dense_nodes)} canonical dense nodes; "
            "provide one count for all dense nodes or one count per dense node in optimization-report order"
        )
    return {node.id: value for node, value in zip(dense_nodes, values)}


def compile_model(model_path: str | Path, output_dir: str | Path, strategy: str = "auto",
                  disabled_passes: set[str] | None = None, model_name: str = "GeneratedModel",
                  max_stack_bytes: int = 65536, team_output_threshold: int = 64,
                  max_team_scratch_bytes: int = 49152, streaming_output_threshold: int = 8,
                  half2_accumulators: int | str | list[int] | tuple[int, ...] | None = None,
                  streaming_recompute_threshold: int = 64) -> dict[str, Any]:
    nonnegative_options = {
        "--max-stack-bytes": max_stack_bytes,
        "--team-output-threshold": team_output_threshold,
        "--max-team-scratch-bytes": max_team_scratch_bytes,
        "--streaming-output-threshold": streaming_output_threshold,
        "--streaming-recompute-threshold": streaming_recompute_threshold,
    }
    for option, value in nonnegative_options.items():
        if value < 0:
            raise CompilerError(f"{option} must be nonnegative; got {value}")
    original, optimized, pass_report = load_and_optimize(model_path, disabled_passes)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schedule = schedule_dense_chains(
        optimized, streaming_output_threshold, streaming_recompute_threshold
    )
    if strategy == "auto":
        maximum_parallel_neurons = max(
            (optimized.tensors[node.outputs[0]].sample_size for node in optimized.nodes),
            default=optimized.tensors[optimized.outputs[0]].sample_size,
        )
        strategy = (
            "sample-local" if schedule.has_streaming
            else ("team" if maximum_parallel_neurons >= team_output_threshold else "sample-local")
        )
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype.value == "float32" else 8
    num_inputs = optimized.tensors[optimized.inputs[0]].sample_size
    num_outputs = optimized.tensors[optimized.outputs[0]].sample_size
    tensorcore_chain = find_tensorcore_dense_chain(optimized)
    tensorcore_scratch_bytes = estimate_tensorcore_scratch_bytes(optimized, tensorcore_chain)
    tensorcore_eligible = (
        tensorcore_chain is not None and scalar_bytes == 4 and num_outputs <= 16 and
        (len(tensorcore_chain) == 3 or num_inputs <= 8) and tensorcore_scratch_bytes <= 49152
    )
    if strategy == "tensorcore" and not tensorcore_eligible:
        raise CompilerError(
            "tensorcore strategy requires a supported float32 two- or three-dense chain with at most 16 outputs; "
            "the two-dense form additionally supports at most 8 inputs, and the three-dense form must require no "
            "more than 49152 bytes of generated shared memory per warp"
        )
    if strategy not in {"sample-local", "team", "tensorcore", "half2"}:
        raise CompilerError(
            f"unknown execution strategy {strategy!r}; choose auto, sample-local, team, tensorcore, or half2"
        )

    plan = plan_storage(optimized)
    sample_plan = plan_storage(
        optimized, schedule.eliminated_tensors,
        schedule.recompute_liveness_extensions(optimized),
    )
    stack_elements = sample_plan.total_elements
    stack_bytes = stack_elements * scalar_bytes
    team_scratch_bytes = plan.total_elements * scalar_bytes
    if stack_bytes > max_stack_bytes:
        raise CompilerError(
            f"planned local activation storage is {stack_bytes} bytes, above --max-stack-bytes={max_stack_bytes}; "
            "use a larger explicit threshold or add a scratch/workspace execution strategy"
        )
    if team_scratch_bytes > max_team_scratch_bytes:
        raise CompilerError(
            f"planned per-team scratch is {team_scratch_bytes} bytes, above "
            f"--max-team-scratch-bytes={max_team_scratch_bytes}; increase the explicit device-specific threshold "
            "or use infer_batch"
        )
    maximum_parallel_neurons = max(
        (optimized.tensors[node.outputs[0]].sample_size for node in optimized.nodes),
        default=optimized.tensors[optimized.outputs[0]].sample_size,
    )
    default_batch_tile, maximum_batch_tile = _hierarchical_batch_tiles(
        maximum_parallel_neurons, team_scratch_bytes, max_team_scratch_bytes
    )
    offsets, manifest = write_weights(optimized, output_path)
    payload_elements = int(manifest["payload_bytes"]) // scalar_bytes
    scalar_code = 1 if scalar_bytes == 4 else 2
    explicit_half2_accumulators = _explicit_half2_accumulator_map(optimized, half2_accumulators)
    header = emit_cpp(
        optimized, plan, sample_plan, schedule, offsets, output_path, model_name,
        strategy, payload_elements, scalar_code,
        default_batch_tile, maximum_batch_tile, streaming_output_threshold,
        explicit_half2_accumulators
    )
    report = _report(
        original, optimized, pass_report, strategy, plan, sample_plan, schedule, scalar_bytes,
        default_batch_tile, maximum_batch_tile, streaming_output_threshold
    )
    report["generated_header"] = header.name
    report["weights"] = "weights.bin"
    report["manifest"] = "weights.json"
    report["auto_strategy_rule"] = (
        "recommend infer_batch when the deterministic dense-chain scheduler selects streaming or recomputation; "
        "otherwise recommend "
        f"infer_batch_hierarchical when an operation has at least {team_output_threshold} output neurons; "
        "the Tensor Core and half2 targets require explicit selection because they change floating-point semantics; "
        "all five inference families are emitted"
    )
    report["half2"] = {
        "batch_lanes": "two adjacent samples",
        "input_output_views": f"{optimized.tensors[optimized.inputs[0]].dtype.value} API boundary",
        "weight_storage": "one persistent scalar FP16 DeviceSpace view; each value is splatted across both lanes",
        "multiply_type": "FP16",
        "accumulator_type": "FP16 partial sums with an FP32 merge for multi-accumulator variants",
        "launch": "Kokkos RangePolicy over ceil(batch_size / 2)",
        "selection": "half2 is explicit-only; infer_batch_half2_heuristic is its default generated target",
        "default_target": "infer_batch_half2_heuristic",
    }
    dense_nodes = [node for node in optimized.nodes if node.op in {"Dense", "DenseBiasActivation"}]
    heuristic_half2_accumulators = half2_accumulator_plan(
        optimized, streaming_output_threshold, schedule
    )
    report["half2"]["heuristic"] = [
        {
            "dense_index": index,
            "node_id": node.id,
            "dot_length": optimized.tensors[node.inputs[0]].sample_size,
            "output_size": optimized.tensors[node.outputs[0]].sample_size,
            "accumulators": heuristic_half2_accumulators[node.id],
        }
        for index, node in enumerate(dense_nodes)
    ]
    if explicit_half2_accumulators is not None:
        report["generated_targets"].append("infer_batch_half2_explicit")
        report["execution_strategies"]["infer_batch_half2_explicit"] = (
            "Kokkos half2 inference using user-specified per-dense accumulator counts"
        )
        report["half2"]["explicit"] = [
            {
                "dense_index": index,
                "node_id": node.id,
                "dot_length": optimized.tensors[node.inputs[0]].sample_size,
                "output_size": optimized.tensors[node.outputs[0]].sample_size,
                "accumulators": explicit_half2_accumulators[node.id],
            }
            for index, node in enumerate(dense_nodes)
        ]
    report["maximum_parallel_neurons"] = maximum_parallel_neurons
    report["streaming_output_threshold"] = streaming_output_threshold
    report["streaming_recompute_threshold"] = streaming_recompute_threshold
    if tensorcore_chain is not None and len(tensorcore_chain) == 3:
        maximum_tensorcore_warps = 1
        while (maximum_tensorcore_warps * 2 <= 8 and
               maximum_tensorcore_warps * 2 * tensorcore_scratch_bytes <= 49152):
            maximum_tensorcore_warps *= 2
        hidden_size = optimized.tensors[tensorcore_chain[-2].outputs[0]].sample_size
        if hidden_size <= 4:
            measured_warps = 4
        elif hidden_size <= 16:
            measured_warps = 2
        elif hidden_size <= 32:
            measured_warps = 4
        elif hidden_size <= 64:
            measured_warps = 2
        else:
            measured_warps = 1
        default_tensorcore_warps = min(maximum_tensorcore_warps, measured_warps)
    else:
        maximum_tensorcore_warps = 8
        hidden_size = (optimized.tensors[tensorcore_chain[0].outputs[0]].sample_size
                       if tensorcore_chain is not None else 0)
        default_tensorcore_warps = 4 if hidden_size <= 16 else (2 if hidden_size <= 256 else 1)
    report["tensorcore"] = {
        "eligible": tensorcore_eligible,
        "batch_tile": 16,
        "launch": "raw CUDA kernel (no Kokkos execution policy)",
        "dense_layers": len(tensorcore_chain) if tensorcore_chain is not None else 0,
        "shared_memory_bytes_per_warp": tensorcore_scratch_bytes,
        "default_warps_per_block": default_tensorcore_warps,
        "maximum_warps_per_block": maximum_tensorcore_warps,
        "input_mode": "TF32",
        "accumulator_type": "float32",
        "selection": "explicit-only",
        "shape_limits": "at most 16 outputs; two-dense form supports at most 8 inputs; "
                        "three-dense form is limited to 49152 generated shared-memory bytes per warp",
    }

    # Deterministic compiler self-check catches invalid fusions independently of ONNX Runtime.
    rng = np.random.default_rng(20260802)
    input_size = optimized.tensors[optimized.inputs[0]].sample_size
    verification_input = rng.standard_normal((input_size, 7)).astype(
        np.float32 if scalar_bytes == 4 else np.float64
    )
    original_output = run_graph(original, verification_input)
    optimized_output = run_graph(optimized, verification_input)
    difference = np.abs(original_output - optimized_output)
    report["ir_optimization_max_absolute_error"] = float(difference.max(initial=0.0))
    tolerance = 2e-6 if scalar_bytes == 4 else 1e-12
    if not np.allclose(original_output, optimized_output, rtol=tolerance, atol=tolerance):
        index = np.unravel_index(int(np.argmax(difference)), difference.shape)
        raise CompilerError(
            f"optimized IR verification failed at output {index[0]}, sample {index[1]}: "
            f"unfused={original_output[index]}, optimized={optimized_output[index]}, "
            f"absolute_error={difference[index]}"
        )
    (output_path / "canonical_ir.json").write_text(optimized.to_json() + "\n")
    (output_path / "optimization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
