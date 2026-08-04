#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class Configuration:
    name: str
    architecture: str
    width: int
    depth: int


CONFIGURATIONS = (
    Configuration("SeqW16D2", "sequential", 16, 2),
    Configuration("SeqW32D2", "sequential", 32, 2),
    Configuration("SeqW64D2", "sequential", 64, 2),
    Configuration("SeqW128D2", "sequential", 128, 2),
    Configuration("SeqW32D8", "sequential", 32, 8),
    Configuration("SeqW64D8", "sequential", 64, 8),
    Configuration("ResidualW32D4", "residual", 32, 4),
    Configuration("ResidualW64D4", "residual", 64, 4),
    Configuration("ResidualW32D8", "residual", 32, 8),
    Configuration("ResidualW64D8", "residual", 64, 8),
    Configuration("ResidualW128D8", "residual", 128, 8),
    Configuration("LongSkipW32D8", "long_skip", 32, 8),
    Configuration("LongSkipW64D8", "long_skip", 64, 8),
    Configuration("Branch4W32D2", "branch4", 32, 2),
    Configuration("Branch4W64D2", "branch4", 64, 2),
)


def parameters(
    rng: np.random.Generator,
    width: int,
    name: str,
    output_width: int | None = None,
) -> list[onnx.TensorProto]:
    output_width = width if output_width is None else output_width
    weight = (rng.standard_normal((width, output_width)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(output_width) * 0.05).astype(np.float32)
    return [
        numpy_helper.from_array(weight, f"{name}_weight"),
        numpy_helper.from_array(bias, f"{name}_bias"),
    ]


def dense(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    rng: np.random.Generator,
    source: str,
    name: str,
    width: int,
    activation: bool = True,
    output_width: int | None = None,
) -> str:
    initializers.extend(parameters(rng, width, name, output_width))
    linear = f"{name}_linear"
    nodes.append(helper.make_node("Gemm", [source, f"{name}_weight", f"{name}_bias"], [linear]))
    if not activation:
        return linear
    output = f"{name}_output"
    nodes.append(helper.make_node("Tanh", [linear], [output]))
    return output


def hidden_graph(
    configuration: Configuration,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    rng: np.random.Generator,
) -> str:
    width = configuration.width
    if configuration.architecture == "sequential":
        current = "input_batch_major"
        for layer in range(configuration.depth):
            current = dense(nodes, initializers, rng, current, f"layer{layer}", width)
        return current

    if configuration.architecture == "residual":
        current = "input_batch_major"
        for block in range(configuration.depth // 2):
            residual = current
            current = dense(nodes, initializers, rng, current, f"block{block}_layer0", width)
            current = dense(
                nodes,
                initializers,
                rng,
                current,
                f"block{block}_layer1",
                width,
                activation=False,
            )
            added = f"block{block}_added"
            output = f"block{block}_output"
            nodes.append(helper.make_node("Add", [current, residual], [added]))
            nodes.append(helper.make_node("Tanh", [added], [output]))
            current = output
        return current

    if configuration.architecture == "long_skip":
        stem = dense(nodes, initializers, rng, "input_batch_major", "stem", width)
        current = stem
        for layer in range(configuration.depth - 1):
            current = dense(nodes, initializers, rng, current, f"layer{layer}", width)
        nodes.append(helper.make_node("Add", [current, stem], ["long_skip_added"]))
        nodes.append(helper.make_node("Tanh", ["long_skip_added"], ["long_skip_output"]))
        return "long_skip_output"

    branches = []
    for branch in range(4):
        current = "input_batch_major"
        for layer in range(configuration.depth):
            current = dense(nodes, initializers, rng, current, f"branch{branch}_layer{layer}", width)
        branches.append(current)
    nodes.append(helper.make_node("Sum", branches, ["branches_joined"]))
    nodes.append(helper.make_node("Tanh", ["branches_joined"], ["branches_output"]))
    return "branches_output"


def create_model(configuration: Configuration, output_path: Path) -> None:
    seed = 20260804 + sum(ord(character) for character in configuration.name)
    rng = np.random.default_rng(seed)
    nodes = [helper.make_node("Transpose", ["input"], ["input_batch_major"], perm=[1, 0])]
    initializers: list[onnx.TensorProto] = []
    hidden = hidden_graph(configuration, nodes, initializers, rng)
    output_batch_major = dense(
        nodes,
        initializers,
        rng,
        hidden,
        "output",
        configuration.width,
        activation=False,
        output_width=3,
    )
    nodes.append(helper.make_node("Transpose", [output_batch_major], ["output"], perm=[1, 0]))
    graph = helper.make_graph(
        nodes,
        configuration.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [configuration.width, "batch"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, "batch"])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="ponni-batch-team-architecture-experiment",
    )
    model.metadata_props.add(key="ponni.orientation", value="features_batch")
    model.metadata_props.add(key="ponni.batch_symbol", value="batch")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for configuration in CONFIGURATIONS:
        create_model(configuration, args.output_dir / f"{configuration.name}.onnx")


if __name__ == "__main__":
    main()
