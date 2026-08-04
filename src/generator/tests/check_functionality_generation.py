#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def dense_count(report: dict[str, object]) -> int:
    operations = report["optimized_operations"]
    assert isinstance(operations, list)
    return sum(operation in {"Dense", "DenseBiasActivation"} for operation in operations)


def check_header(path: Path) -> None:
    generated = path.read_text()
    direct_batch = generated.split("void infer_batch_half2(", 1)[0].rsplit("void infer_batch(", 1)[1]
    if direct_batch.count("input_view(i,ibatch)") != 1:
        raise RuntimeError(f"{path.name}: direct View kernel must stage each input exactly once")
    for forbidden in ("inputs(j,ibatch)", "input_view(j,ibatch)", "Kokkos::View<"):
        if forbidden in direct_batch:
            raise RuntimeError(f"{path.name}: direct View kernel contains forbidden reread/storage {forbidden!r}")
    if "preactivation" in generated:
        raise RuntimeError(f"{path.name}: generated dense preactivation was materialized")
    for target in ("infer_one", "infer_batch", "infer_batch_half2"):
        if f"void {target}(" not in generated:
            raise RuntimeError(f"{path.name}: missing generated target {target}")
    for forbidden in ("infer_batch_hierarchical", "infer_batch_team", "Kokkos::TeamPolicy", "team_shmem"):
        if forbidden in generated:
            raise RuntimeError(f"{path.name}: obsolete generated construct {forbidden!r} remains")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--resnet", type=Path, required=True)
    parser.add_argument("--densenet", type=Path, required=True)
    parser.add_argument("--branching", type=Path, required=True)
    parser.add_argument("--operator-zoo", type=Path, required=True)
    parser.add_argument("--workspace-levels", type=Path, required=True)
    args = parser.parse_args()

    reports: dict[str, dict[str, object]] = {}
    model_directories = {
        name: directory for name, directory in vars(args).items() if name != "workspace_levels"
    }
    for name, directory in model_directories.items():
        check_header(next(directory.glob("*.hpp")))
        reports[name] = json.loads((directory / "optimization_report.json").read_text())
        if reports[name]["storage"]["external_workspace_bytes"] != 0:
            raise RuntimeError(f"{name}: generated inference unexpectedly requires external activation workspace")

    if dense_count(reports["deep"]) != 10:
        raise RuntimeError("depth-10 model did not retain ten canonical dense operations")
    deep_local = reports["deep"]["sample_local_storage"]
    deep_slots = deep_local["plan"]["slots"].values()
    required_extent = max((slot["offset"] + slot["size"] for slot in deep_slots), default=0)
    if deep_local["workspace_elements"] != required_extent:
        raise RuntimeError("varying-width dense-chain storage does not cover the complete live workspace extent")
    if deep_local["streamed_dense_pairs"] < 2:
        raise RuntimeError("depth-10 model did not exercise generalized non-overlapping dense-chain streaming")
    if dense_count(reports["resnet"]) != 10:
        raise RuntimeError("depth-10 ResNet did not retain ten canonical dense operations")
    if reports["resnet"]["optimized_operations"].count("ResidualAddActivation") != 5:
        raise RuntimeError("depth-10 ResNet did not fuse its five residual activations")
    if dense_count(reports["densenet"]) != 5 or "Concat" not in reports["densenet"]["optimized_operations"]:
        raise RuntimeError("DenseNet did not retain dense connectivity through static feature concatenation")
    branch_counts = reports["branching"]["operator_counts"]
    for activation in ("Relu", "Sigmoid", "Tanh"):
        if int(branch_counts.get(activation, 0)) == 0:
            raise RuntimeError(f"branching model does not exercise {activation}")
    if reports["branching"]["dense_chain_schedule"]["decision_counts"]["retain"] < 1:
        raise RuntimeError("branching model did not retain its shared activation")

    level_reports = [
        json.loads((args.workspace_levels / f"level_{level}" / "optimization_report.json").read_text())
        for level in range(1, 6)
    ]
    for level, report in enumerate(level_reports, 1):
        if report["workspace_reduction_aggressiveness"] != level:
            raise RuntimeError(f"workspace level {level} report records the wrong policy")
        check_header(next((args.workspace_levels / f"level_{level}").glob("*.hpp")))
    counts = [report["dense_chain_schedule"]["decision_counts"] for report in level_reports]
    if counts[0]["stream"] != 0 or counts[0]["recompute"] != 0:
        raise RuntimeError("workspace level 1 unexpectedly streams or recomputes")
    if not 0 < counts[1]["stream"] < counts[2]["stream"]:
        raise RuntimeError("workspace levels 2 and 3 did not progressively enable dense streaming")
    if counts[2]["recompute"] != 0:
        raise RuntimeError("workspace level 3 unexpectedly recomputes a shared branch")
    if counts[3]["recompute"] != 1:
        raise RuntimeError("workspace level 4 did not recompute exactly the two-consumer branch")
    if counts[4]["recompute"] != 2:
        raise RuntimeError("workspace level 5 did not additionally recompute the three-consumer branch")
    zoo_operations = set(reports["operator_zoo"]["optimized_operations"])
    required = {
        "BatchNormalization", "LayerNormalization", "Softmax", "LogSoftmax", "ReduceMean", "ReduceSum",
        "LeakyRelu", "Elu", "Gelu", "Softplus", "HardSigmoid", "HardSwish", "Mish", "Silu", "Clip",
        "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Cos", "Cosh", "Erf", "Floor", "Round",
        "Sign", "Sin", "Sinh", "Tan",
        "And", "Cast", "CompareSelect", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Not", "Or",
        "Where", "Xor", "Celu", "Gather", "IsInf", "IsNaN", "LpNormalization", "Mean", "PRelu", "ReduceL1",
        "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceSumSquare",
        "Selu", "Softsign", "Sum", "ThresholdedRelu",
    }
    missing = required - zoo_operations
    if missing:
        raise RuntimeError(f"operator zoo is missing optimized operations: {sorted(missing)}")
    zoo_header = next(args.operator_zoo.glob("*.hpp")).read_text()
    for forbidden in ("preactivation", "Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace> workspace"):
        if forbidden in zoo_header:
            raise RuntimeError(f"operator zoo emitted forbidden tensor temporary {forbidden!r}")


if __name__ == "__main__":
    main()
