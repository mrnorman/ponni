#!/usr/bin/env python3
"""Export PONNI examples across CPU framework stacks and compare their ONNX.

The top-level mode creates isolated uv environments, invokes the worker mode in
each environment, and then analyzes every emitted model with one designated
PONNI/ONNX installation. Results include exact package provenance, exporter
logs, raw ONNX summaries, canonical/optimized operation sequences, storage
plans, and numerical errors measured by the existing exporters.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parents[3]
GENERATOR = REPOSITORY / "src" / "generator"
DEFAULT_MATRIX = Path(__file__).with_name("onnx_version_matrix.json")
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TRACKED_PACKAGES = (
    "numpy", "torch", "keras", "tensorflow", "tensorflow-cpu", "tf2onnx",
    "onnx", "onnxruntime", "onnxscript", "onnx-ir",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _all_package_versions() -> dict[str, str]:
    return dict(sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ))


def _exception() -> dict[str, str]:
    exception = sys.exception()
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc(),
    }


def _export_worker(output_dir: Path) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "environment": {
            "packages": _package_versions(),
            "all_packages": _all_package_versions(),
            "python": sys.version,
            "platform": platform.platform(),
            "keras_backend": None,
        },
        "models": {},
    }

    def record(name: str, exporter: Callable[[], Any]) -> None:
        try:
            result = exporter()
            report["models"][name] = {
                "status": "exported",
                "model": str(result.model_path.relative_to(output_dir)),
                "reference": str(result.reference_path.relative_to(output_dir)),
                "max_onnx_absolute_error": result.max_onnx_absolute_error,
                "max_onnx_relative_error": result.max_onnx_relative_error,
            }
        except Exception:
            report["models"][name] = {"status": "failed", "error": _exception()}

    try:
        import torch
        from kokkos_nn.export import export_module, make_example_models, make_functionality_models

        mlp, residual = make_example_models()
        record("pytorch_mlp", lambda: export_module(mlp, 4, output_dir, "pytorch_mlp", batch_sizes=(1, 3, 7)))
        record(
            "pytorch_residual",
            lambda: export_module(residual, 4, output_dir, "pytorch_residual", batch_sizes=(1, 3, 7)),
        )
        for index, (name, (model, width)) in enumerate(make_functionality_models().items()):
            record(
                f"pytorch_{name}",
                lambda name=name, model=model, width=width, index=index: export_module(
                    model, width, output_dir, f"pytorch_{name}", batch_sizes=(1, 3, 7), seed=8200 + index
                ),
            )
        report["environment"]["torch_cpu_build"] = torch.version.cuda is None and torch.version.hip is None
    except Exception:
        report["pytorch_setup_error"] = _exception()

    try:
        import keras
        import tensorflow as tf
        from kokkos_nn.framework_export import (
            export_keras_model,
            export_keras_normalization_model,
            export_tensorflow_model,
        )

        report["environment"]["keras_backend"] = keras.backend.backend()
        report["environment"]["tensorflow_devices"] = [device.device_type for device in tf.config.list_physical_devices()]
        tensorflow_build = tf.sysconfig.get_build_info()
        report["environment"]["tensorflow_cpu_build"] = not (
            tensorflow_build.get("is_cuda_build", False) or tensorflow_build.get("is_rocm_build", False)
        )
        record("keras_mlp", lambda: export_keras_model(output_dir, batch_sizes=(1, 3, 7)))
        record("keras_normalization", lambda: export_keras_normalization_model(output_dir, batch_sizes=(1, 3, 7)))
        record("tensorflow_residual", lambda: export_tensorflow_model(output_dir, batch_sizes=(1, 3, 7)))
    except Exception:
        report["framework_setup_error"] = _exception()

    _write_json(output_dir / "export_report.json", report)
    return 0


def _raw_onnx_summary(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path)
    node_types = [node.op_type for node in model.graph.node]
    domains = sorted({node.domain or "ai.onnx" for node in model.graph.node})
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "node_count": len(node_types),
        "node_types": node_types,
        "operator_counts": dict(sorted(Counter(node_types).items())),
        "domains": domains,
        "initializer_count": len(model.graph.initializer),
        "constant_node_count": node_types.count("Constant"),
    }


def _analysis_worker(root: Path) -> int:
    import onnx
    from kokkos_nn.compiler import validate_model

    provenance_path = root / "provenance.json"
    selected = set(json.loads(provenance_path.read_text())["selected_stacks"]) if provenance_path.is_file() else None
    stack_directories = [
        path for path in sorted((root / "stacks").iterdir())
        if selected is None or path.name in selected
    ]
    for stack_dir in stack_directories:
        export_report_path = stack_dir / "models" / "export_report.json"
        if not export_report_path.is_file():
            continue
        export_report = json.loads(export_report_path.read_text())
        analysis: dict[str, Any] = {
            "analysis_environment": _package_versions(),
            "models": {},
        }
        for name, export in sorted(export_report.get("models", {}).items()):
            if export.get("status") != "exported":
                analysis["models"][name] = {"status": "not-exported"}
                continue
            model_path = stack_dir / "models" / export["model"]
            entry: dict[str, Any] = {"raw_onnx": _raw_onnx_summary(model_path)}
            try:
                report = validate_model(model_path)
                entry.update({
                    "status": "accepted",
                    "canonical_operations": report["canonical_operations"],
                    "optimized_operations": report["optimized_operations"],
                    "storage": report["storage"],
                    "sample_local_storage": report["sample_local_storage"],
                    "dense_chain_schedule": report["dense_chain_schedule"],
                    "ir_optimization_max_absolute_error": report.get("ir_optimization_max_absolute_error"),
                })
            except Exception:
                entry.update({"status": "rejected", "error": _exception()})
                message = entry["error"]["message"]
                if "dynamic non-batch dimension" in message and "only the batch dimension may be dynamic" in message:
                    model = onnx.load(model_path)
                    dynamic_dimensions = [
                        dimension.dim_param
                        for value in (model.graph.input[0], model.graph.output[0])
                        for dimension in value.type.tensor_type.shape.dim
                        if dimension.dim_param
                    ]
                    if dynamic_dimensions and len(set(dynamic_dimensions)) == 1:
                        actual_symbol = dynamic_dimensions[0]
                        metadata = {item.key: item for item in model.metadata_props}
                        if "ponni.batch_symbol" in metadata:
                            metadata["ponni.batch_symbol"].value = actual_symbol
                        else:
                            item = model.metadata_props.add()
                            item.key = "ponni.batch_symbol"
                            item.value = actual_symbol
                        retry_path = stack_dir / "compatibility" / f"{name}-actual-batch-symbol.onnx"
                        retry_path.parent.mkdir(parents=True, exist_ok=True)
                        onnx.save_model(model, retry_path, save_as_external_data=False)
                        try:
                            retry = validate_model(retry_path)
                            entry["batch_symbol_retry"] = {
                                "status": "accepted",
                                "actual_symbol": actual_symbol,
                                "canonical_operations": retry["canonical_operations"],
                                "optimized_operations": retry["optimized_operations"],
                                "storage": retry["storage"],
                                "sample_local_storage": retry["sample_local_storage"],
                                "dense_chain_schedule": retry["dense_chain_schedule"],
                            }
                        except Exception:
                            entry["batch_symbol_retry"] = {
                                "status": "rejected",
                                "actual_symbol": actual_symbol,
                                "error": _exception(),
                            }
            analysis["models"][name] = entry
        _write_json(stack_dir / "analysis.json", analysis)
    _write_comparison(root)
    return 0


def _write_comparison(root: Path) -> None:
    provenance_path = root / "provenance.json"
    selected = set(json.loads(provenance_path.read_text())["selected_stacks"]) if provenance_path.is_file() else None
    stacks: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "stacks").glob("*/analysis.json")):
        if selected is not None and path.parent.name not in selected:
            continue
        stacks[path.parent.name] = json.loads(path.read_text())
    model_names = sorted({name for report in stacks.values() for name in report["models"]})
    comparison: dict[str, Any] = {"stacks": sorted(stacks), "models": {}}
    for name in model_names:
        entries = {stack: report["models"].get(name, {"status": "missing"}) for stack, report in stacks.items()}
        variants: dict[str, list[str]] = {}
        for field in ("raw_onnx", "canonical_operations", "optimized_operations", "planning"):
            groups: dict[str, list[str]] = {}
            for stack, entry in entries.items():
                effective = entry.get("batch_symbol_retry", entry)
                value = effective.get(field)
                if field == "raw_onnx" and value is not None:
                    value = entry.get(field)
                    value = {
                        key: value[key]
                        for key in ("ir_version", "opsets", "node_types", "initializer_count", "constant_node_count")
                    }
                elif field == "planning" and effective.get("status") == "accepted":
                    value = {
                        "storage": effective["storage"],
                        "sample_local_storage": effective["sample_local_storage"],
                        "dense_chain_schedule": effective["dense_chain_schedule"],
                    }
                key = json.dumps(value, sort_keys=True)
                groups.setdefault(key, []).append(stack)
            variants[field] = [", ".join(group) for group in groups.values()]
        comparison["models"][name] = {
            "status": {stack: entry.get("status", "missing") for stack, entry in entries.items()},
            "raw_variant_count": len(variants["raw_onnx"]),
            "canonical_variant_count": len(variants["canonical_operations"]),
            "optimized_variant_count": len(variants["optimized_operations"]),
            "planning_variant_count": len(variants["planning"]),
            "variant_groups": variants,
        }
    _write_json(root / "comparison.json", comparison)


def _run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        process = subprocess.run(command, cwd=REPOSITORY, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")


def _resolve_python(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY / candidate


def _orchestrate(args: argparse.Namespace) -> int:
    matrix = json.loads(args.matrix.read_text())
    selected = set(args.stacks or [stack["name"] for stack in matrix["stacks"]])
    stacks = [stack for stack in matrix["stacks"] if stack["name"] in selected]
    missing = selected - {stack["name"] for stack in stacks}
    if missing:
        raise ValueError(f"unknown stacks: {', '.join(sorted(missing))}")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    except Exception:
        revision = "unknown"
    _write_json(root / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "git_revision": revision,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrix_path": str(args.matrix.resolve()),
        "matrix": matrix,
        "selected_stacks": [stack["name"] for stack in stacks],
        "command": sys.argv,
    })
    uv = args.uv or "uv"
    worker = str(Path(__file__).resolve())
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(GENERATOR), "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"})
    completed: list[str] = []
    for stack in stacks:
        stack_dir = root / "stacks" / stack["name"]
        models_dir = stack_dir / "models"
        stack_dir.mkdir(parents=True, exist_ok=True)
        try:
            if "python_path" in stack:
                python = _resolve_python(stack["python_path"])
            else:
                environment_dir = stack_dir / "environment"
                python = environment_dir / "bin" / "python"
                if not python.is_file():
                    _run_logged(
                        [uv, "venv", "--python", stack["python"], str(environment_dir)],
                        stack_dir / "create-environment.log",
                    )
                    _run_logged(
                        [uv, "pip", "install", "--python", str(python), *stack["packages"]],
                        stack_dir / "install-frameworks.log",
                    )
                    _run_logged(
                        [uv, "pip", "install", "--python", str(python), "--index-url", CPU_TORCH_INDEX,
                         f"torch=={stack['torch']}"],
                        stack_dir / "install-pytorch.log",
                    )
            if not python.is_file():
                raise FileNotFoundError(f"Python executable does not exist: {python}")
            _run_logged(
                [str(python), worker, "--worker", "export", "--output-dir", str(models_dir)],
                stack_dir / "export.log", environment,
            )
            completed.append(stack["name"])
        except Exception:
            _write_json(stack_dir / "orchestration_error.json", _exception())
    analysis_python = _resolve_python(args.analysis_python)
    _run_logged(
        [str(analysis_python), worker, "--worker", "analyze", "--output-dir", str(root)],
        root / "analysis.log", environment,
    )
    print(json.dumps({"output_dir": str(root), "completed_stacks": completed}, indent=2))
    return 0 if len(completed) == len(stacks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stacks", nargs="*")
    parser.add_argument("--analysis-python", default="unit/build/python_env/bin/python")
    parser.add_argument("--uv")
    parser.add_argument("--worker", choices=("export", "analyze"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == "export":
        return _export_worker(args.output_dir)
    if args.worker == "analyze":
        return _analysis_worker(args.output_dir)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
