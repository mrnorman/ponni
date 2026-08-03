#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <generator_gpu_scale> <weight.bin>..." >&2
  exit 2
fi

output_file="$(mktemp /tmp/ponni_gpu_scale.XXXXXX)"
trap 'rm -f "${output_file}"' EXIT

set +e
"$@" >"${output_file}" 2>&1
benchmark_status=$?
set -e

if [[ ${benchmark_status} -ne 0 ]]; then
  echo "generator_gpu_scale failed with exit status ${benchmark_status}; raw output follows:" >&2
  cat "${output_file}" >&2
  exit "${benchmark_status}"
fi

awk '
function value(name,    i, prefix) {
  prefix = name "="
  for (i = 2; i <= NF; i++) {
    if (index($i, prefix) == 1) return substr($i, length(prefix) + 1)
  }
  return "-"
}

$1 == "generator_gpu_summary" {
  summary_count++
  summary_rows = summary_rows sprintf("| %s | %s | %s | %s | %s | %s (%s) | %s | %s (%s) | %s | %s |\n",
      value("width"), value("batch"), value("sarray_ms"), value("view_batch_ms"),
      value("hierarchical_tile1_ms"), value("hierarchical_best_ms"),
      value("hierarchical_best_tile"), value("half2_ms"), value("half2_best_ms"),
      value("half2_best_policy"), value("half2_best_error"),
      value("half2_most_accurate_policy"))
  next
}

$1 == "generator_gpu_half2_policy" {
  half2_rows = half2_rows sprintf("| %s | %s | %s | %s | %s |\n",
      value("width"), value("batch"), value("policy"), value("half2_ms"),
      value("max_abs_difference"))
  next
}

$1 == "generator_gpu_tile" {
  tile_rows = tile_rows sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |\n",
      value("width"), value("batch"), value("tile"), value("default_tile"),
      value("tiled_ms"), value("speedup"), value("max_abs_difference"),
      value("sarray_max_abs_difference"))
  next
}

NF != 0 {
  other_output = other_output $0 "\n"
}

END {
  if (summary_count == 0) {
    print "ERROR: generator_gpu_scale produced no summary records." > "/dev/stderr"
    if (other_output != "") print other_output > "/dev/stderr"
    exit 3
  }

  print ""
  print "PONNI generator GPU performance summary"
  print "======================================="
  print "Times are milliseconds per inference call. Lower is better."
  print ""
  print "| Width | Batch | SArray ms | View ms | Hier tile 1 ms | Best hier ms (tile) | Half2 baseline ms | Best half2 ms (policy) | Minimum half2 error | Most accurate half2 |"
  print "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"
  printf "%s", summary_rows

  print ""
  print "Half2 accumulator policies"
  print "--------------------------"
  print "| Width | Batch | Policy | Time ms | Max abs difference from View |"
  print "|---:|---:|:---|---:|---:|"
  printf "%s", half2_rows

  print ""
  print "Hierarchical batch tiles"
  print "------------------------"
  print "| Width | Batch | Tile | Generated default | Time ms | Speedup vs View | Max abs difference | SArray max abs difference |"
  print "|---:|---:|---:|---:|---:|---:|---:|---:|"
  printf "%s", tile_rows

  if (other_output != "") {
    print ""
    print "Additional benchmark output"
    print "---------------------------"
    printf "%s", other_output
  }
}
' "${output_file}"
