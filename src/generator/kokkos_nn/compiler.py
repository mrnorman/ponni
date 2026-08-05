"""Coordinate ONNX import, optimization, planning, verification, and emission.

This module is intentionally orchestration-heavy. Semantic decisions belong in
the importer and passes; storage decisions belong in the planner and scheduler;
and C++ spelling belongs in the emitter. The public functions here assemble
those phases and produce a stable, inspectable report.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .emitter import emit_cpp
from .errors import CompilerError
from .importer import import_onnx
from .interpreter import run_graph
from .ir import DType
from .passes import optimize
from .planner import plan_storage
from .scheduler import DenseChainSchedule, schedule_dense_chains
from .weights import write_weights


def _onnxscript_preprocess(model_path: str | Path):
    """Apply the optional provider-neutral ONNX cleanup before PONNI import."""
    try:
        import onnx
        from onnxscript import optimizer

        model = onnx.load(Path(model_path))
        before = model.SerializeToString()
        optimized = optimizer.optimize(model)
        return optimized, optimized.SerializeToString() != before
    except ImportError as exc:
        raise CompilerError(
            "ONNX Script preprocessing requested, but onnxscript is not installed"
        ) from exc
    except Exception as exc:
        raise CompilerError(f"ONNX Script preprocessing failed for {model_path}: {exc}") from exc


def load_and_optimize(model_path: str | Path, disabled_passes: set[str] | None = None,
                      onnx_preprocess: bool = False):
    """Import both the source contract and the graph selected for optimization."""
    # Preserve a direct import of the source model for reporting and for the
    # final equivalence check. Optional preprocessing is never allowed to hide
    # what the user supplied at the PONNI boundary.
    original = import_onnx(model_path)
    preprocessor_changed = False
    if onnx_preprocess:
        preprocessed_model, preprocessor_changed = _onnxscript_preprocess(model_path)
        canonical = import_onnx(preprocessed_model)
    else:
        canonical = original
    optimized, pass_report = optimize(canonical, disabled_passes)
    pass_report.insert(0, {
        "name": "onnxscript-preprocess",
        "disabled": not onnx_preprocess,
        "changed": preprocessor_changed,
        "nodes_before": len(original.nodes),
        "nodes_after": len(canonical.nodes),
    })
    return original, optimized, pass_report


def _shape_string(shape: tuple[object, ...]) -> str:
    return "(" + ", ".join(getattr(dim, "name", str(dim)) for dim in shape) + ")"


def _fusion_rejections(graph, schedule: DenseChainSchedule) -> list[str]:
    """Explain materialized dense outputs that might otherwise look unfused."""
    graph.rebuild_links()
    reasons: list[str] = []
    for node in graph.nodes:
        if node.op not in {"Dense", "DenseBiasActivation", "DenseEpilogue", "DenseResidualActivation", "Add"}:
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


def _optimized_component_operations(graph) -> list[str]:
    """Flatten nested fused programs so reports retain their source operations."""
    operations: list[str] = []

    def record_steps(steps) -> None:
        for step in steps:
            operations.append(str(step["op"]))
            record_steps(step.get("attributes", {}).get("steps", []))

    for node in graph.nodes:
        operations.append(node.op)
        record_steps(node.attributes.get("steps", []))
        record_steps(node.attributes.get("map_steps", []))
        record_steps(node.attributes.get("map_region_steps", []))
        record_steps(node.attributes.get("epilogue_steps", []))
    return operations


def _report(original, optimized, pass_report, sample_plan, sample_mask_plan,
            schedule: DenseChainSchedule, scalar_bytes: int,
            workspace_oracle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the machine-readable contract emitted by validate and compile."""
    input_tensor = original.tensors[original.inputs[0]]
    output_tensor = original.tensors[original.outputs[0]]
    learned_parameter_count = sum(
        constant.values.size for constant in optimized.constants.values() if constant.learned
    )
    canonical_counts = dict(sorted(Counter(node.op for node in optimized.nodes).items()))
    fused_ops = sum(
        count for op, count in canonical_counts.items()
        if op in {"CompareSelect", "DenseBiasActivation", "DenseEpilogue", "DenseResidualActivation", "ElementwiseChain",
                  "PointwiseRegion", "ResidualAddActivation"}
    )
    storage_report = sample_plan.to_dict(scalar_bytes)
    storage_report["mask_plan"] = sample_mask_plan.to_dict(1)
    storage_report["estimated_stack_bytes"] += sample_mask_plan.total_elements
    report = {
        "model_inputs": [{"name": input_tensor.name, "shape": _shape_string(input_tensor.shape),
                          "dtype": input_tensor.dtype.value}],
        "model_outputs": [{"name": output_tensor.name, "shape": _shape_string(output_tensor.shape),
                           "dtype": output_tensor.dtype.value}],
        "onnx_ir_version": original.metadata.get("ir_version"),
        "onnx_opsets": original.metadata.get("opsets", {}),
        "onnx_operator_schema_counts": original.metadata.get("operator_schema_counts", {}),
        "operator_counts": original.metadata.get("operator_counts", {}),
        "learned_parameter_count": int(learned_parameter_count),
        "original_onnx_node_count": int(original.metadata.get("original_node_count", len(original.nodes))),
        "canonical_node_count": len(original.nodes),
        "fused_node_count": len(optimized.nodes),
        "fused_operation_count": fused_ops,
        "canonical_operations": [node.op for node in original.nodes],
        "optimized_operations": [node.op for node in optimized.nodes],
        "optimized_component_operations": _optimized_component_operations(optimized),
        "passes": pass_report,
        "storage": storage_report,
        "dense_chain_schedule": schedule.to_dict(),
        "workspace_reduction_aggressiveness": schedule.aggressiveness,
        "sample_local_storage": {
            "workspace_elements": sample_plan.total_elements,
            "workspace_bytes": sample_plan.total_elements * scalar_bytes,
            "mask_workspace_elements": sample_mask_plan.total_elements,
            "mask_workspace_bytes": sample_mask_plan.total_elements,
            "plan": sample_plan.to_dict(scalar_bytes),
            "mask_plan": sample_mask_plan.to_dict(1),
            "batch_input_staging_elements": optimized.tensors[optimized.inputs[0]].sample_size,
            "batch_input_staging_bytes": optimized.tensors[optimized.inputs[0]].sample_size * scalar_bytes,
            "streamed_dense_pairs": sum(
                decision.action == "stream" for decision in schedule.decisions.values()
            ),
            "recomputed_activations": sum(
                decision.action == "recompute" for decision in schedule.decisions.values()
            ),
            "extra_recomputation_madds": schedule.recompute_extra_madds,
        },
        "generated_targets": ["infer_one", "infer_batch", "infer_batch_half2"],
        "execution_strategies": {
            "infer_one": "device-inline fixed SArray inference with planned local storage",
            "infer_batch": "Kokkos RangePolicy over samples with planned local storage",
            "infer_batch_half2": (
                "Kokkos RangePolicy over adjacent sample pairs using ponni::TwoHalf and one dependent "
                "FP16 accumulation chain per dense dot product"
            ),
        },
        "half2": {
            "batch_lanes": "two adjacent samples",
            "input_output_views": f"{optimized.tensors[optimized.inputs[0]].dtype.value} API boundary",
            "weight_storage": "persistent scalar FP16 model-memory view, splatted across both lanes",
            "multiply_type": "FP16",
            "accumulator_type": "one dependent FP16 chain",
            "launch": "Kokkos RangePolicy over ceil(batch_size / 2)",
        },
        "fusion_rejections": _fusion_rejections(optimized, schedule),
    }
    if workspace_oracle is not None:
        report["workspace_oracle"] = workspace_oracle
    return report


def _plan_comparison(native_plan, heuristic_plan, exact_plan) -> dict[str, Any]:
    return {
        "heuristic_elements": heuristic_plan.total_elements,
        "native_elements": native_plan.total_elements,
        "exact_elements": exact_plan.total_elements,
        "native_saved_elements": heuristic_plan.total_elements - native_plan.total_elements,
        "heuristic_optimality_gap": heuristic_plan.total_elements - exact_plan.total_elements,
        "exact_backend": exact_plan.placement_strategy,
        "optimality_proven": exact_plan.optimality_proven,
    }


def _plans(optimized, workspace_reduction_aggressiveness: int,
           analyze_workspace: bool = False):
    """Create the execution schedule and separate floating/Boolean arenas."""
    # Boolean intermediates use byte storage, while floating intermediates use
    # the model scalar type. They therefore need independent liveness arenas.
    schedule = schedule_dense_chains(optimized, workspace_reduction_aggressiveness)
    floating = {DType.FLOAT32, DType.FLOAT64}
    sample_plan = plan_storage(
        optimized, schedule.eliminated_tensors,
        schedule.recompute_liveness_extensions(optimized), floating,
    )
    sample_mask_plan = plan_storage(optimized, dtypes={DType.BOOL})
    oracle = None
    if analyze_workspace:
        # The oracle is diagnostic only. Native plans still use the bounded,
        # deterministic strategy selected by plan_storage above.
        floating_heuristic = plan_storage(
            optimized, schedule.eliminated_tensors,
            schedule.recompute_liveness_extensions(optimized), floating,
            placement="heuristic",
        )
        floating_exact = plan_storage(
            optimized, schedule.eliminated_tensors,
            schedule.recompute_liveness_extensions(optimized), floating,
            placement="exact",
        )
        mask_heuristic = plan_storage(optimized, dtypes={DType.BOOL}, placement="heuristic")
        mask_exact = plan_storage(optimized, dtypes={DType.BOOL}, placement="exact")
        oracle = {
            "scope": "arena placement for the selected fusion/streaming/recomputation schedule",
            "floating": _plan_comparison(sample_plan, floating_heuristic, floating_exact),
            "boolean": _plan_comparison(sample_mask_plan, mask_heuristic, mask_exact),
        }
    return schedule, sample_plan, sample_mask_plan, oracle


def _validate_workspace_reduction_aggressiveness(value: int) -> None:
    if value not in range(1, 6):
        raise CompilerError(
            "--workspace-reduction-aggressiveness must be an integer from 1 through 5; "
            f"got {value}"
        )


def validate_model(model_path: str | Path, disabled_passes: set[str] | None = None,
                   workspace_reduction_aggressiveness: int = 3,
                   onnx_preprocess: bool = False,
                   analyze_workspace: bool = False) -> dict[str, Any]:
    """Validate a model and return the same analysis used by compilation."""
    _validate_workspace_reduction_aggressiveness(workspace_reduction_aggressiveness)
    original, optimized, pass_report = load_and_optimize(
        model_path, disabled_passes, onnx_preprocess,
    )
    schedule, sample_plan, sample_mask_plan, oracle = _plans(
        optimized, workspace_reduction_aggressiveness, analyze_workspace,
    )
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype == DType.FLOAT32 else 8
    return _report(
        original, optimized, pass_report, sample_plan, sample_mask_plan, schedule, scalar_bytes, oracle,
    )


def compile_model(model_path: str | Path, output_dir: str | Path,
                  disabled_passes: set[str] | None = None,
                  model_name: str = "GeneratedModel",
                  workspace_reduction_aggressiveness: int = 3,
                  onnx_preprocess: bool = False,
                  analyze_workspace: bool = False) -> dict[str, Any]:
    """Compile one ONNX model into a header, weights, canonical IR, and report."""
    _validate_workspace_reduction_aggressiveness(workspace_reduction_aggressiveness)
    original, optimized, pass_report = load_and_optimize(
        model_path, disabled_passes, onnx_preprocess,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schedule, sample_plan, sample_mask_plan, oracle = _plans(
        optimized, workspace_reduction_aggressiveness, analyze_workspace,
    )
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype == DType.FLOAT32 else 8
    offsets, manifest = write_weights(optimized, output_path)
    payload_elements = int(manifest["payload_bytes"]) // scalar_bytes
    scalar_code = 1 if scalar_bytes == 4 else 2
    header = emit_cpp(
        optimized, sample_plan, sample_mask_plan, schedule, offsets, output_path, model_name,
        payload_elements, scalar_code,
    )
    report = _report(
        original, optimized, pass_report, sample_plan, sample_mask_plan, schedule, scalar_bytes, oracle,
    )
    report["generated_header"] = header.name
    report["weights"] = "weights.ponni"
    report["manifest"] = "weights.json"

    # Compare source and optimized IRs before committing the report. The fixed
    # seed keeps failures reproducible and covers several batch samples at once.
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
    (output_path / "optimization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
