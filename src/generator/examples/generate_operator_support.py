#!/usr/bin/env python3
"""Generate the user-facing operator table from ONNX and importer schemas."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import onnx

from kokkos_nn.importer import (
    MAX_SUPPORTED_ONNX_OPSET,
    MIN_SUPPORTED_ONNX_OPSET,
    SUPPORTED_OPERATOR_SCHEMAS,
)


DOCUMENT_PATH = Path(__file__).resolve().parents[1] / "ONNX_OPERATOR_SUPPORT.md"
ONNX_OPERATOR_URL = "https://onnx.ai/onnx/operators/onnx__{name}.html"

# Restrictions live next to the document generator rather than being parsed
# from implementation text. The schema registry remains authoritative for what
# is accepted; these strings explain the narrower PONNI contract to users.
OPERATOR_RESTRICTIONS = {
    "And": "Boolean inputs and output; only scalar and exact per-sample shape broadcasting.",
    "BatchNormalization": "Five-input inference mode; parameters must be constant scalars or feature-sized tensors.",
    "Cast": "Boolean-to-float32/float64 conversion only; elementwise and shape-preserving.",
    "CastLike": "No-op same-type casts and Boolean-to-float32/float64 conversion only.",
    "Clip": "Minimum and maximum, when present, must be compile-time scalar constants.",
    "Concat": "Rank-two tensors; concatenation is limited to the static feature axis.",
    "Constant": "Tensor-valued bool, float32, float64, int32, and int64 constants.",
    "Equal": "Matching float or Boolean inputs and Boolean output; only scalar and exact-shape broadcasting.",
    "Dropout": "Inference mode and primary output only; canonicalized to Identity with no runtime work.",
    "Flatten": "Must preserve per-sample element count and ordering.",
    "Gelu": "The `approximate` attribute must be `none` or `tanh`.",
    "Gemm": "Constant rank-two weights, `transA=0`, and scalar or output-vector bias.",
    "Gather": "Compile-time scalar/vector indices on the static feature axis, or a statically foldable input.",
    "IsInf": "Elementwise floating input and Boolean output; honors positive/negative detection attributes.",
    "IsNaN": "Elementwise floating input and Boolean output.",
    "LayerNormalization": "Complete static feature axis; constant scale/bias; `stash_type=1`.",
    "LogSoftmax": "Rank-two tensors; complete static feature axis only.",
    "LpNormalization": "Complete static feature axis with p=1 or p=2.",
    "MatMul": "Right operand must be a constant rank-two weight.",
    "Not": "Boolean input and output; elementwise and shape-preserving.",
    "Or": "Boolean inputs and output; only scalar and exact per-sample shape broadcasting.",
    "PRelu": "Floating data with scalar or exact per-sample slope broadcasting.",
    "ReduceMean": "Complete static feature axis; constant axes; `keepdims=1`.",
    "ReduceSum": "Complete static feature axis; constant axes; `keepdims=0` or `keepdims=1`.",
    "Reshape": "Compile-time shape, `allowzero=0`, and unchanged per-sample element order.",
    "Shape": "Compile-time evaluation only; selected dimensions may not depend on runtime batch size.",
    "Size": "Compile-time evaluation only; input dimensions may not depend on runtime batch size.",
    "Softmax": "Rank-two tensors; complete static feature axis only.",
    "Transpose": "Batch-axis movement or constant-weight transpose only.",
    "Squeeze": "Constant axes; must preserve the batch axis, sample element count, and element order.",
    "Unsqueeze": "Constant axes; must preserve the batch axis, sample element count, and element order.",
    "Where": "Boolean condition, matching floating branches/output, and scalar or exact-shape broadcasting.",
    "Xor": "Boolean inputs and output; only scalar and exact per-sample shape broadcasting.",
}

for reduction in (
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin",
    "ReduceProd", "ReduceSumSquare",
):
    OPERATOR_RESTRICTIONS[reduction] = (
        "Complete static feature axis; constant axes; `keepdims=0` or `keepdims=1`."
    )

for comparison in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
    OPERATOR_RESTRICTIONS[comparison] = (
        "Matching floating inputs and Boolean output; only scalar and exact per-sample shape broadcasting."
    )

BINARY_OPERATORS = {"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"}
VARIADIC_OPERATORS = {"Mean", "Sum"}
UNARY_OPERATORS = {
    "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos", "Cosh", "Elu", "Erf",
    "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "LeakyRelu", "Log", "Mish", "Neg", "Reciprocal",
    "Relu", "Round", "Selu", "Sigmoid", "Sign", "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Tan", "Tanh",
    "ThresholdedRelu",
}


def _schemas_reached_by_supported_opsets() -> dict[str, set[int]]:
    """Resolve the schema version each accepted opset selects per operator."""
    histories = defaultdict(list)
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain == "":
            histories[schema.name].append(schema)

    reached: dict[str, set[int]] = {}
    for name, schemas in histories.items():
        versions: set[int] = set()
        for opset in range(MIN_SUPPORTED_ONNX_OPSET, MAX_SUPPORTED_ONNX_OPSET + 1):
            candidates = [schema.since_version for schema in schemas if schema.since_version <= opset]
            if candidates:
                versions.add(max(candidates))
        if versions:
            reached[name] = versions
    return reached


def _restriction(name: str) -> str:
    if name in OPERATOR_RESTRICTIONS:
        return OPERATOR_RESTRICTIONS[name]
    if name in BINARY_OPERATORS:
        return "Only scalar and exact per-sample shape broadcasting."
    if name in VARIADIC_OPERATORS:
        return "One or more matching floating inputs; only scalar and exact per-sample shape broadcasting."
    if name in UNARY_OPERATORS:
        return "Elementwise; must preserve the per-sample shape."
    return "—"


def _status(name: str, reached: set[int]) -> str:
    reviewed = SUPPORTED_OPERATOR_SCHEMAS.get(name, set())
    if not reviewed:
        return "Unsupported"
    if reached <= reviewed:
        return "Supported with restrictions" if _restriction(name) != "—" else "Supported"
    if reached & reviewed:
        return "Some schemas supported"
    return "Unsupported"


def _versions(values: set[int]) -> str:
    return ", ".join(str(value) for value in sorted(values)) or "—"


def _escape(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ")


def render_document() -> str:
    """Render a complete deterministic Markdown support matrix."""
    reached = _schemas_reached_by_supported_opsets()
    unknown = sorted(set(SUPPORTED_OPERATOR_SCHEMAS) - set(reached))
    if unknown:
        raise RuntimeError(
            "PONNI's schema registry contains operators absent from the installed ONNX definitions: "
            + ", ".join(unknown)
        )

    rows = []
    counts = defaultdict(int)
    for name in sorted(reached, key=str.casefold):
        status = _status(name, reached[name])
        counts[status] += 1
        reviewed = SUPPORTED_OPERATOR_SCHEMAS.get(name, set())
        link = f"[{name}]({ONNX_OPERATOR_URL.format(name=name)})"
        rows.append(
            f"| {link} | {_versions(reached[name])} | {status} | {_versions(reviewed)} | "
            f"{_escape(_restriction(name) if reviewed else '—')} |"
        )

    supported = sum(count for status, count in counts.items() if status != "Unsupported")
    unsupported = counts["Unsupported"]
    return "\n".join([
        "# ONNX operator support",
        "",
        "<!-- Generated by examples/generate_operator_support.py; do not edit by hand. -->",
        "",
        f"PONNI accepts standard `ai.onnx` opsets {MIN_SUPPORTED_ONNX_OPSET} through "
        f"{MAX_SUPPORTED_ONNX_OPSET}. This table contains every standard-domain operator whose schema can be selected "
        "by an opset in that envelope. Operators introduced only in later opsets are outside this compatibility "
        "contract and are listed by the [current ONNX operator index](https://onnx.ai/onnx/operators/index.html).",
        "",
        f"Within this envelope, PONNI accepts **{supported}** operator names in at least one documented, "
        f"potentially restricted form; the remaining **{unsupported}** operator names are not accepted. "
        "“Supported” means that every schema selected in the envelope has been reviewed. "
        "“Some schemas supported” means only the versions in the PONNI schema column are accepted. Operator-specific "
        "restrictions below apply in addition to PONNI's fixed-feature, single-input/single-output model contract.",
        "",
        "| Operator | ONNX schemas selected in opsets 13–22 | PONNI status | PONNI-reviewed schemas | Restrictions |",
        "|---|---:|---|---:|---|",
        *rows,
        "",
        "## Regenerating this page",
        "",
        "Run from the repository root with the generator environment active:",
        "",
        "```bash",
        "PYTHONPATH=src/generator python src/generator/examples/generate_operator_support.py",
        "```",
        "",
        "Use `--check` to verify that the checked-in page matches the importer registry and installed ONNX schemas.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PONNI's ONNX operator-support table")
    parser.add_argument("--output", type=Path, default=DOCUMENT_PATH)
    parser.add_argument("--check", action="store_true", help="fail instead of writing when the output is stale")
    args = parser.parse_args()
    rendered = render_document()
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(f"{args.output} is stale; regenerate it with {Path(__file__).name}", file=sys.stderr)
            raise SystemExit(1)
        return
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
