#!/usr/bin/env python3
"""Create deterministic feature-major ONNX MLPs for GPU scheduling benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


WIDTHS = (4, 8, 16, 32, 64, 128)


def create_model(width: int, output_path: Path) -> None:
    rng = np.random.default_rng(20260802 + width)
    weight0 = (rng.standard_normal((width, width)) * 0.2).astype(np.float32)
    bias0 = (rng.standard_normal(width) * 0.1).astype(np.float32)
    weight1 = (rng.standard_normal((width, width)) * 0.2).astype(np.float32)
    bias1 = (rng.standard_normal(width) * 0.1).astype(np.float32)
    weight2 = (rng.standard_normal((width, 3)) * 0.2).astype(np.float32)
    bias2 = (rng.standard_normal(3) * 0.1).astype(np.float32)

    nodes = [
        helper.make_node("Transpose", ["input"], ["input_batch_major"], perm=[1, 0]),
        helper.make_node("Gemm", ["input_batch_major", "weight0", "bias0"], ["hidden_linear"]),
        helper.make_node("Tanh", ["hidden_linear"], ["hidden"]),
        helper.make_node("Gemm", ["hidden", "weight1", "bias1"], ["hidden_linear_1"]),
        helper.make_node("Tanh", ["hidden_linear_1"], ["hidden_1"]),
        helper.make_node("Gemm", ["hidden_1", "weight2", "bias2"], ["output_batch_major"]),
        helper.make_node("Transpose", ["output_batch_major"], ["output"], perm=[1, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        f"ponni_width_{width}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [width, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, "batch"])],
        [
            numpy_helper.from_array(weight0, "weight0"),
            numpy_helper.from_array(bias0, "bias0"),
            numpy_helper.from_array(weight1, "weight1"),
            numpy_helper.from_array(bias1, "bias1"),
            numpy_helper.from_array(weight2, "weight2"),
            numpy_helper.from_array(bias2, "bias2"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="ponni-kokkos-nn-benchmark",
    )
    model.metadata_props.add(key="ponni.orientation", value="features_batch")
    model.metadata_props.add(key="ponni.batch_symbol", value="batch")
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for width in WIDTHS:
        create_model(width, args.output_dir / f"width_{width}.onnx")


if __name__ == "__main__":
    main()
