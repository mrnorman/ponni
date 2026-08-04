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
            reasons.append(
                f"node {node.id} ({node.op}) retains its output because it has {len(consumers)} consumers"
                + (f": {decision.reason}" if decision is not None else "")
            )
    return reasons


def _report(original, optimized, pass_report, sample_plan, sample_mask_plan,
            schedule: DenseChainSchedule, scalar_bytes: int) -> dict[str, Any]:
    input_tensor = original.tensors[original.inputs[0]]
    output_tensor = original.tensors[original.outputs[0]]
    learned_parameter_count = sum(
        constant.values.size for constant in optimized.constants.values() if constant.learned
    )
    canonical_counts = dict(sorted(Counter(node.op for node in optimized.nodes).items()))
    fused_ops = sum(
        count for op, count in canonical_counts.items()
        if op in {"CompareSelect", "DenseBiasActivation", "ElementwiseChain", "ResidualAddActivation"}
    )
    storage_report = sample_plan.to_dict(scalar_bytes)
    storage_report["mask_plan"] = sample_mask_plan.to_dict(1)
    storage_report["estimated_stack_bytes"] += sample_mask_plan.total_elements
    return {
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
        "passes": pass_report,
        "storage": storage_report,
        "dense_chain_schedule": schedule.to_dict(),
        "sample_local_storage": {
            "workspace_elements": sample_plan.total_elements,
            "workspace_bytes": sample_plan.total_elements * scalar_bytes,
            "mask_workspace_elements": sample_mask_plan.total_elements,
            "mask_workspace_bytes": sample_mask_plan.total_elements,
            "plan": sample_plan.to_dict(scalar_bytes),
            "mask_plan": sample_mask_plan.to_dict(1),
            "batch_input_staging_elements": optimized.tensors[optimized.inputs[0]].sample_size,
            "batch_input_staging_bytes": optimized.tensors[optimized.inputs[0]].sample_size * scalar_bytes,
            "streamed_dense_pairs": len(schedule.pair_by_consumer),
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
            "weight_storage": "persistent scalar FP16 DeviceSpace view, splatted across both lanes",
            "multiply_type": "FP16",
            "accumulator_type": "one dependent FP16 chain",
            "launch": "Kokkos RangePolicy over ceil(batch_size / 2)",
        },
        "batch_fastest": True,
        "fusion_rejections": _fusion_rejections(optimized, schedule),
        "rejected_constructs": [],
    }


def _plans(optimized):
    schedule = schedule_dense_chains(optimized)
    floating = {DType.FLOAT32, DType.FLOAT64}
    sample_plan = plan_storage(optimized, schedule.eliminated_tensors, dtypes=floating)
    sample_mask_plan = plan_storage(optimized, dtypes={DType.BOOL})
    return schedule, sample_plan, sample_mask_plan


def validate_model(model_path: str | Path, disabled_passes: set[str] | None = None) -> dict[str, Any]:
    original, optimized, pass_report = load_and_optimize(model_path, disabled_passes)
    schedule, sample_plan, sample_mask_plan = _plans(optimized)
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype == DType.FLOAT32 else 8
    return _report(
        original, optimized, pass_report, sample_plan, sample_mask_plan, schedule, scalar_bytes,
    )


def compile_model(model_path: str | Path, output_dir: str | Path,
                  disabled_passes: set[str] | None = None,
                  model_name: str = "GeneratedModel") -> dict[str, Any]:
    original, optimized, pass_report = load_and_optimize(model_path, disabled_passes)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schedule, sample_plan, sample_mask_plan = _plans(optimized)
    scalar_bytes = 4 if optimized.tensors[optimized.inputs[0]].dtype == DType.FLOAT32 else 8
    offsets, manifest = write_weights(optimized, output_path)
    payload_elements = int(manifest["payload_bytes"]) // scalar_bytes
    scalar_code = 1 if scalar_bytes == 4 else 2
    header = emit_cpp(
        optimized, sample_plan, sample_mask_plan, schedule, offsets, output_path, model_name,
        payload_elements, scalar_code,
    )
    report = _report(
        original, optimized, pass_report, sample_plan, sample_mask_plan, schedule, scalar_bytes,
    )
    report["generated_header"] = header.name
    report["weights"] = "weights.bin"
    report["manifest"] = "weights.json"

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
