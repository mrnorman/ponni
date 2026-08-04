#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Framework export is a CPU build-time activity, even when generated inference targets a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from kokkos_nn.framework_export import export_keras_model, export_keras_normalization_model, export_tensorflow_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Keras and TensorFlow PONNI examples to ONNX")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true", help="do not print the JSON report to stdout")
    args = parser.parse_args()
    results = {
        "keras_mlp": export_keras_model(args.output_dir),
        "keras_normalization": export_keras_normalization_model(args.output_dir),
        "tensorflow_residual": export_tensorflow_model(args.output_dir),
    }
    summary = {
        name: {
            "model": str(result.model_path),
            "reference": str(result.reference_path),
            "max_onnx_absolute_error": result.max_onnx_absolute_error,
            "max_onnx_relative_error": result.max_onnx_relative_error,
        }
        for name, result in results.items()
    }
    (args.output_dir / "framework_export_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
