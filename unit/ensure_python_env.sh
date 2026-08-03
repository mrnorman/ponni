#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <uv_path_file> <python_executable> <venv_dir>" >&2
  exit 2
fi

uv_path_file="$1"
python_exe="$2"
venv_dir="$3"
venv_python="${venv_dir}/bin/python"

if [[ ! -f "${uv_path_file}" ]]; then
  echo "ERROR: uv path file not found: ${uv_path_file}" >&2
  exit 1
fi

uv_bin="$(cat "${uv_path_file}")"
if [[ ! -x "${uv_bin}" ]]; then
  echo "ERROR: uv executable is not valid: ${uv_bin}" >&2
  exit 1
fi

want_version="$("${python_exe}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

env_is_valid=false
if [[ -x "${venv_python}" ]]; then
  have_version="$("${venv_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [[ "${have_version}" == "${want_version}" ]] && \
     "${venv_python}" -c 'import torch, keras, h5py, numpy, onnx, onnxruntime, onnxscript' >/dev/null 2>&1; then
    env_is_valid=true
  fi
fi

if [[ "${env_is_valid}" != "true" ]]; then
  rm -rf "${venv_dir}"
  "${uv_bin}" venv --python "${python_exe}" "${venv_dir}"
  "${uv_bin}" pip install --python "${venv_python}" numpy h5py torch keras onnx onnxruntime onnxscript
fi

rm -f "${venv_dir}/.gitignore"
