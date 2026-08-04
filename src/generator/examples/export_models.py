#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kokkos_nn.export import export_module, export_operator_zoo, make_example_models, make_functionality_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic PONNI generator example models")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true", help="do not print the JSON report to stdout")
    args = parser.parse_args()
    mlp, residual = make_example_models()
    results = {
        "mlp": export_module(mlp, 4, args.output_dir, "mlp"),
        "residual": export_module(residual, 4, args.output_dir, "residual"),
    }
    for name, (model, num_inputs) in make_functionality_models().items():
        results[name] = export_module(
            model, num_inputs, args.output_dir, name, batch_sizes=(1, 2, 3, 7, 11), seed=8128 + len(results)
        )
    results["operator_zoo"] = export_operator_zoo(args.output_dir)
    summary = {
        name: {
            "model": str(result.model_path),
            "reference": str(result.reference_path),
            "max_onnx_absolute_error": result.max_onnx_absolute_error,
            "max_onnx_relative_error": result.max_onnx_relative_error,
        }
        for name, result in results.items()
    }
    (args.output_dir / "export_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
