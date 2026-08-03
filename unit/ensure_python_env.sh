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
export UV_CACHE_DIR="$(dirname "${python_install_dir}")/cache"
export UV_NO_CONFIG=1
export PYTHONHOME=
export PYTHONNOUSERSITE=1
if [[ "${backend}" != "CUDA" ]]; then
  export CUDA_VISIBLE_DEVICES=-1
  export JAX_PLATFORMS=cpu
  export TF_CPP_MIN_LOG_LEVEL=3
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
import os
import tensorflow as tf
import torch
backend = os.environ["PONNI_EXPECTED_BACKEND"]
distributions = {name.lower().replace("_", "-") for name in
                 metadata.packages_distributions().get("tensorflow", [])}
build = tf.sysconfig.get_build_info()
if backend == "CPU":
    assert torch.version.cuda is None and torch.version.hip is None
    assert "tensorflow-cpu" in distributions
    assert not build.get("is_cuda_build", False) and not build.get("is_rocm_build", False)
elif backend == "CUDA":
    assert torch.version.cuda is not None
    assert "tensorflow" in distributions
    assert build.get("is_cuda_build", False)
elif backend == "HIP":
    assert torch.version.hip is not None
    assert "tensorflow-rocm" in distributions
    assert build.get("is_rocm_build", False)
else:
    raise AssertionError(f"unsupported backend {backend}")'

frameworks_ready=false
if PONNI_EXPECTED_BACKEND="${backend}" "${venv_python}" -c \
     'import keras, tf2onnx, h5py, numpy, onnx, onnxruntime, onnxscript' >/dev/null 2>&1 && \
   PONNI_EXPECTED_BACKEND="${backend}" "${venv_python}" -c "${backend_check}" >/dev/null 2>&1 && \
   "${uv_bin}" pip check --python "${venv_python}" >/dev/null 2>&1; then
  frameworks_ready=true
fi

if [[ "${frameworks_ready}" != "true" ]]; then
  case "${backend}" in
    CPU)
      "${uv_bin}" pip install --python "${venv_python}" tensorflow-cpu
      "${uv_bin}" pip install --python "${venv_python}" \
        --index-url https://download.pytorch.org/whl/cpu torch
      ;;
    CUDA)
      "${uv_bin}" pip install --python "${venv_python}" 'tensorflow[and-cuda]' torch
      ;;
    HIP)
      if [[ -z "${ROCM_PATH:-}" ]]; then
        echo "ERROR: ROCM_PATH is required to install HIP framework packages" >&2
        exit 1
      fi
      rocm_version="${ROCM_PATH##*-}"
      rocm_major_minor="${rocm_version%.*}"
      pytorch_index="${PONNI_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm${rocm_version}}"
      tensorflow_links="${PONNI_TENSORFLOW_FIND_LINKS:-https://repo.radeon.com/rocm/manylinux/rocm-rel-${rocm_major_minor}/}"
      tensorflow_requirement="${PONNI_TENSORFLOW_REQUIREMENT:-tensorflow-rocm}"
      if [[ "${rocm_major_minor}" == "6.2" ]] && [[ -z "${PONNI_TENSORFLOW_REQUIREMENT:-}" ]]; then
        tensorflow_requirement='tensorflow-rocm==2.16.1'
      fi
      "${uv_bin}" pip install --python "${venv_python}" \
        --index-url "${pytorch_index}" torch
      "${uv_bin}" pip install --python "${venv_python}" \
        --find-links "${tensorflow_links}" "${tensorflow_requirement}"
      ;;
    SYCL)
      echo "ERROR: SYCL framework packages require explicit Intel package sources" >&2
      exit 1
      ;;
    *)
      echo "ERROR: unsupported framework backend '${backend}'" >&2
      exit 1
      ;;
  esac

  common_requirements=('numpy<2' h5py keras tf2onnx onnxruntime)
  if [[ "${backend}" == "HIP" ]]; then
    common_requirements+=(
      'ml-dtypes~=0.3.1'
      'onnx>=1.16,<1.18'
      'onnxscript==0.3.2'
      'onnx-ir==0.1.3'
    )
  else
    common_requirements+=(onnx onnxscript)
  fi
  "${uv_bin}" pip install --python "${venv_python}" "${common_requirements[@]}"
fi

if ! PONNI_EXPECTED_BACKEND="${backend}" "${venv_python}" -c \
       'import keras, tf2onnx, h5py, numpy, onnx, onnxruntime, onnxscript' >/dev/null 2>&1 || \
   ! PONNI_EXPECTED_BACKEND="${backend}" "${venv_python}" -c "${backend_check}" >/dev/null 2>&1 || \
   ! "${uv_bin}" pip check --python "${venv_python}" >/dev/null 2>&1; then
  echo "ERROR: python_env packages do not match the requested ${backend} backend" >&2
  echo "Replace python_env only with explicit user permission" >&2
  exit 1
fi

printf "%s\n" "${backend}" > "${backend_marker}"
rm -f "${venv_dir}/.gitignore"
touch "${venv_dir}/.ready"
