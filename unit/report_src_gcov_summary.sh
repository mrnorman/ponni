#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-$(pwd)}"
GCOV_BIN="${GCOV_BIN:-gcov}"

if ! command -v "${GCOV_BIN}" >/dev/null 2>&1; then
  echo "ERROR: gcov executable not found: ${GCOV_BIN}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
OUT_DIR="${BUILD_DIR}/coverage"
RAW_FILE="${OUT_DIR}/gcov_intermediate_all.txt"
SUMMARY_FILE="${OUT_DIR}/src_gcov_summary.txt"
TMP_COVERED="${OUT_DIR}/src_covered.tmp"
TMP_ALL="${OUT_DIR}/src_all.tmp"

mkdir -p "${OUT_DIR}"
: > "${RAW_FILE}"

mapfile -d '' gcov_inputs < <(find "${BUILD_DIR}" -type f -name '*.gcda' ! -path '*/kokkos/*' -print0)

if [[ ${#gcov_inputs[@]} -eq 0 ]]; then
  echo "NOTE: no .gcda files found; falling back to .gcno (results may show 0.00% before tests run)" >> "${RAW_FILE}"
  mapfile -d '' gcov_inputs < <(find "${BUILD_DIR}" -type f -name '*.gcno' ! -path '*/kokkos/*' -print0)
fi

for gcov_input in "${gcov_inputs[@]}"; do
  echo "### GCOV_INPUT ${gcov_input}" >> "${RAW_FILE}"
  "${GCOV_BIN}" -i "${gcov_input}" >> "${RAW_FILE}" 2>&1 || true
  echo >> "${RAW_FILE}"
done

awk -v src_dir="${SRC_DIR}" '
  /^File '\''/ {
    file = $0
    sub(/^File '\''/, "", file)
    sub(/'\''$/, "", file)
    current = file
    next
  }
  /^Lines executed:/ {
    if (index(current, src_dir "/") == 1) {
      line = $0
      sub(/^Lines executed:/, "", line)
      split(line, a, "% of ")
      pct = a[1] + 0
      lines = a[2] + 0
      if (!(current in best) || pct > best[current]) {
        best[current] = pct
        line_count[current] = lines
      }
    }
  }
  END {
    for (f in best) {
      printf "%s|%.2f|%d\n", f, best[f], line_count[f]
    }
  }
' "${RAW_FILE}" | sort > "${TMP_COVERED}"

find "${SRC_DIR}" -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) | sort > "${TMP_ALL}"

total_src_files=$(wc -l < "${TMP_ALL}")
covered_src_files=$(wc -l < "${TMP_COVERED}")

{
  echo "total_src_files=${total_src_files}"
  echo "src_files_with_gcov_data=${covered_src_files}"
  echo "src_files_missing_from_gcov=$((total_src_files-covered_src_files))"
  echo
  echo "[covered]"
  if [[ -s "${TMP_COVERED}" ]]; then
    while IFS='|' read -r f pct lines; do
      printf "%6.2f%% of %4d lines :: %s\n" "${pct}" "${lines}" "${f}"
    done < "${TMP_COVERED}"
  fi
  echo
  echo "[missing]"
  awk -F'|' 'NR==FNR{seen[$1]=1; next} !($0 in seen){print $0}' "${TMP_COVERED}" "${TMP_ALL}"
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"
