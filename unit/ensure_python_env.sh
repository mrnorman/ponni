#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <uv_path_file> <python_version> <python_install_dir> <venv_dir> <backend>" >&2
  exit 2
fi

uv_path_file="$1"
python_version="$2"
python_install_dir="$3"
venv_dir="$4"
backend="$5"
venv_python="${venv_dir}/bin/python"
backend_marker="${venv_dir}/.ponni-backend"
onnxruntime_requirement='onnxruntime>=1.25,<2'
export UV_CACHE_DIR="$(dirname "${python_install_dir}")/cache"
export UV_NO_CONFIG=1
export PYTHONHOME=
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export TF_CPP_MIN_LOG_LEVEL=3
if [[ "${backend}" != "CPU" ]]; then
  echo "ERROR: PONNI framework export tests require the unified CPU backend, not ${backend}" >&2
  exit 1
fi

if [[ ! -f "${uv_path_file}" ]]; then
  echo "ERROR: uv path file not found: ${uv_path_file}" >&2
  exit 1
fi

uv_bin="$(<"${uv_path_file}")"
if [[ ! -x "${uv_bin}" ]] || ! "${uv_bin}" --version >/dev/null 2>&1; then
  echo "ERROR: PONNI-owned uv is not valid: ${uv_bin}" >&2
  exit 1
fi

mkdir -p "${python_install_dir}"
managed_python="$(UV_PYTHON_INSTALL_DIR="${python_install_dir}" \
  "${uv_bin}" python find --python-preference only-managed "${python_version}" 2>/dev/null || true)"
if [[ -z "${managed_python}" ]] || [[ ! -x "${managed_python}" ]]; then
  UV_PYTHON_INSTALL_DIR="${python_install_dir}" \
    "${uv_bin}" python install "${python_version}"
  managed_python="$(UV_PYTHON_INSTALL_DIR="${python_install_dir}" \
    "${uv_bin}" python find --python-preference only-managed "${python_version}")"
fi

if [[ ! -x "${venv_python}" ]]; then
  UV_PYTHON_INSTALL_DIR="${python_install_dir}" \
    "${uv_bin}" venv --python "${managed_python}" "${venv_dir}"
fi

have_version="$("${venv_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${have_version}" != "${python_version}" ]]; then
  echo "ERROR: ${venv_python} is Python ${have_version}, expected ${python_version}" >&2
  echo "Replace python_env only with explicit user permission" >&2
  exit 1
fi

if [[ -f "${backend_marker}" ]] && [[ "$(<"${backend_marker}")" != "${backend}" ]]; then
  echo "ERROR: python_env was prepared for $(<"${backend_marker}"), not ${backend}" >&2
  echo "Replace python_env only with explicit user permission" >&2
  exit 1
fi

backend_check='import importlib.metadata as metadata
import tensorflow as tf
import torch
distributions = {name.lower().replace("_", "-") for name in
                 metadata.packages_distributions().get("tensorflow", [])}
build = tf.sysconfig.get_build_info()
assert torch.version.cuda is None and torch.version.hip is None
assert "tensorflow-cpu" in distributions
assert not build.get("is_cuda_build", False) and not build.get("is_rocm_build", False)'

onnxruntime_check='import onnxruntime as ort
parts = ort.__version__.split(".")
version = tuple(int(part) for part in parts[:2])
assert (1, 25) <= version < (2, 0)
assert "CPUExecutionProvider" in ort.get_available_providers()'

# Keep a reused build-local environment within PONNI's reviewed ONNX Runtime
# envelope without reinstalling the larger framework packages.
if ! "${venv_python}" -c "${onnxruntime_check}" >/dev/null 2>&1; then
  "${uv_bin}" pip install --python "${venv_python}" --upgrade 'numpy>=2.0.2' "${onnxruntime_requirement}"
fi

frameworks_ready=false
if "${venv_python}" -c \
     'import flax, jax, keras, sklearn, tf2onnx, numpy, onnx, onnxruntime, onnxscript, safetensors' \
     >/dev/null 2>&1 && \
   "${venv_python}" -c "${backend_check}" >/dev/null 2>&1 && \
   "${venv_python}" -c "${onnxruntime_check}" >/dev/null 2>&1 && \
   "${uv_bin}" pip check --python "${venv_python}" >/dev/null 2>&1; then
  frameworks_ready=true
fi

if [[ "${frameworks_ready}" != "true" ]]; then
  "${uv_bin}" pip install --python "${venv_python}" tensorflow-cpu
  "${uv_bin}" pip install --python "${venv_python}" \
    --index-url https://download.pytorch.org/whl/cpu torch

  common_requirements=(
    'numpy>=2.0.2' flax jax keras scikit-learn tf2onnx
    "${onnxruntime_requirement}" onnx onnxscript safetensors
  )
  "${uv_bin}" pip install --python "${venv_python}" "${common_requirements[@]}"
fi

if ! "${venv_python}" -c \
       'import flax, jax, keras, sklearn, tf2onnx, numpy, onnx, onnxruntime, onnxscript, safetensors' \
       >/dev/null 2>&1 || \
   ! "${venv_python}" -c "${backend_check}" >/dev/null 2>&1 || \
   ! "${venv_python}" -c "${onnxruntime_check}" >/dev/null 2>&1 || \
   ! "${uv_bin}" pip check --python "${venv_python}" >/dev/null 2>&1; then
  echo "ERROR: python_env packages do not match the required CPU framework backend" >&2
  echo "Replace python_env only with explicit user permission" >&2
  exit 1
fi

printf "%s\n" "${backend}" > "${backend_marker}"
rm -f "${venv_dir}/.gitignore"
touch "${venv_dir}/.ready"
