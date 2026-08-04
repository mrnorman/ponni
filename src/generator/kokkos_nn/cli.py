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
    validate.add_argument(
        "--workspace-reduction-aggressiveness", type=int, choices=range(1, 6), default=3,
        help="cross-layer streaming and one-hop recomputation level (1-5; default: 3)",
    )
    validate.add_argument("--quiet", action="store_true", help="do not print the JSON report to stdout")

    compile_command = subparsers.add_parser("compile", help="generate Kokkos C++ and a weight blob")
    compile_command.add_argument("model", type=Path)
    compile_command.add_argument("--output-dir", type=Path, required=True)
    compile_command.add_argument("--model-name", default="GeneratedModel")
    compile_command.add_argument("--disable-pass", action="append", default=[], metavar="NAME[,NAME]")
    compile_command.add_argument(
        "--workspace-reduction-aggressiveness", type=int, choices=range(1, 6), default=3,
        help="cross-layer streaming and one-hop recomputation level (1-5; default: 3)",
    )
    compile_command.add_argument("--quiet", action="store_true", help="do not print the JSON report to stdout")

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
            report = validate_model(
                args.model, disabled, args.workspace_reduction_aggressiveness,
            )
        else:
            report = compile_model(
                args.model,
                args.output_dir,
                disabled,
                args.model_name,
                args.workspace_reduction_aggressiveness,
            )
        if not args.quiet:
            print(json.dumps(report, indent=2, sort_keys=True))
    except CompilerError as exc:
        print(f"kokkos_nn: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
