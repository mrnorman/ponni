#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_record(line: str) -> dict[str, str] | None:
    fields = line.split()
    if not fields or fields[0] != "architecture_result":
        return None
    return dict(field.split("=", 1) for field in fields[1:] if "=" in field)


def planner_candidates(root: Path, name: str) -> dict[int, dict[str, object]]:
    report_path = root / "generated" / name / "optimization_report.json"
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    return {
        int(candidate["team_size"]): candidate
        for candidate in report["batch_team"]["candidates"]
    }


def number(record: dict[str, str], field: str) -> float:
    return float(record[field])


def print_summary(records: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in records:
        grouped.setdefault(record["name"], []).append(record)

    print("\nPONNI batch-team architecture experiment")
    print("========================================")
    print("Batch size is 1,000,000; times are milliseconds per inference call. Lower is better.\n")
    print("| Model | Architecture | Width | Depth | Batch ms | Best team | Batch-team ms | Speedup |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|")
    for name, model_records in grouped.items():
        supported = [record for record in model_records if record["supported"] == "1"]
        best = min(supported, key=lambda record: number(record, "batch_team_ms"))
        print(
            f"| {name} | {best['architecture']} | {best['width']} | {best['depth']} "
            f"| {number(best, 'batch_ms'):.3f} | {best['team_size']} "
            f"| {number(best, 'batch_team_ms'):.3f} | {number(best, 'speedup'):.3f} |"
        )


def print_details(root: Path, records: list[dict[str, str]]) -> None:
    candidate_cache: dict[str, dict[int, dict[str, object]]] = {}
    print("\nAll fixed-team candidates")
    print("-------------------------")
    print(
        "| Model | Team | Time ms | Speedup | Local B/sample | Scratch B/sample "
        "| Scratch B/team | Scratch budget B/team | Max abs difference |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for record in records:
        if record["supported"] != "1":
            continue
        name = record["name"]
        if name not in candidate_cache:
            candidate_cache[name] = planner_candidates(root, name)
        candidate = candidate_cache[name][int(record["team_size"])]
        print(
            f"| {name} | {record['team_size']} | {number(record, 'batch_team_ms'):.3f} "
            f"| {number(record, 'speedup'):.3f} | {candidate['local_bytes_per_sample']} "
            f"| {candidate['scratch_bytes_per_sample']} | {candidate['scratch_bytes_per_team']} "
            f"| {candidate['scratch_budget_bytes_per_team']} "
            f"| {number(record, 'max_abs_difference'):.3e} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    result = subprocess.run(
        [str(args.executable), str(args.root)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    records = [record for line in result.stdout.splitlines() if (record := parse_record(line))]
    if result.returncode != 0 or not records:
        sys.stderr.write(result.stdout)
        if not records:
            sys.stderr.write("ERROR: architecture benchmark produced no result records.\n")
        return result.returncode if result.returncode != 0 else 3

    print_summary(records)
    print_details(args.root, records)
    other_output = [line for line in result.stdout.splitlines() if parse_record(line) is None and line]
    if other_output:
        print("\nAdditional benchmark output")
        print("---------------------------")
        print("\n".join(other_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
