from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compiler import compile_model, validate_model
from .errors import CompilerError
from .passes import PASS_PIPELINE
from .weights import validate_weight_blob


def _disabled(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kokkos_nn", description="Compile fixed-shape ONNX inference DAGs to Kokkos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate, canonicalize, and report a model")
    validate.add_argument("model", type=Path)
    validate.add_argument("--disable-pass", action="append", default=[], metavar="NAME[,NAME]")

    compile_command = subparsers.add_parser("compile", help="generate Kokkos C++ and a weight blob")
    compile_command.add_argument("model", type=Path)
    compile_command.add_argument("--output-dir", type=Path, required=True)
    compile_command.add_argument(
        "--strategy", choices=("auto", "sample-local", "team", "tensorcore", "half2"), default="auto"
    )
    compile_command.add_argument("--model-name", default="GeneratedModel")
    compile_command.add_argument("--disable-pass", action="append", default=[], metavar="NAME[,NAME]")
    compile_command.add_argument("--max-stack-bytes", type=int, default=65536)
    compile_command.add_argument("--max-team-scratch-bytes", type=int, default=49152)
    compile_command.add_argument("--team-output-threshold", type=int, default=64)
    compile_command.add_argument("--streaming-output-threshold", type=int, default=8)
    compile_command.add_argument(
        "--streaming-recompute-threshold",
        type=int,
        default=64,
        metavar="MADDS",
        help="maximum duplicated dense multiply-adds allowed for deterministic terminal-branch recomputation",
    )
    compile_command.add_argument(
        "--half2-accumulators",
        metavar="COUNT[,COUNT...]",
        help=(
            "emit infer_batch_half2_explicit with one accumulator count for every dense node, or one count per "
            "canonical dense node in optimization-report order; supported counts: 0,2,4,8,16,32"
        ),
    )

    passes = subparsers.add_parser("list-passes", help="list deterministic optimization pass names")
    passes.set_defaults(list_passes=True)

    weights = subparsers.add_parser("validate-weights", help="validate a generated binary weight blob")
    weights.add_argument("weights", type=Path)
    weights.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-passes":
            for name, _ in PASS_PIPELINE:
                print(name)
            return
        if args.command == "validate-weights":
            manifest = json.loads(args.manifest.read_text()) if args.manifest else None
            print(json.dumps(validate_weight_blob(args.weights, manifest), indent=2, sort_keys=True))
            return
        disabled = _disabled(args.disable_pass)
        if args.command == "validate":
            report = validate_model(args.model, disabled)
        else:
            report = compile_model(
                args.model,
                args.output_dir,
                args.strategy,
                disabled,
                args.model_name,
                args.max_stack_bytes,
                args.team_output_threshold,
                args.max_team_scratch_bytes,
                args.streaming_output_threshold,
                args.half2_accumulators,
                args.streaming_recompute_threshold,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
    except CompilerError as exc:
        print(f"kokkos_nn: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
