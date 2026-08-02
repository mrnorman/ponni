#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <uv_path_file> <python_executable> <uv_install_dir> [uv_hint]" >&2
  exit 2
fi

uv_path_file="$1"
python_exe="$2"
uv_install_dir="$3"
uv_hint="${4:-}"
local_uv_bin="${uv_install_dir}/bin/uv"

uv_bin=""

mkdir -p "${uv_install_dir}/bin"

if [[ -x "${local_uv_bin}" ]]; then
  uv_bin="${local_uv_bin}"
else
  if [[ -n "${uv_hint}" && -x "${uv_hint}" ]]; then
    cp -f "${uv_hint}" "${local_uv_bin}"
    chmod +x "${local_uv_bin}"
  elif command -v uv >/dev/null 2>&1; then
    cp -f "$(command -v uv)" "${local_uv_bin}"
    chmod +x "${local_uv_bin}"
  else
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${uv_install_dir}/bin" sh
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${uv_install_dir}/bin" sh
    else
      echo "ERROR: uv not found and neither curl nor wget is available to install it locally" >&2
      exit 1
    fi
  fi
  uv_bin="${local_uv_bin}"
fi

if [[ ! -x "${uv_bin}" ]]; then
  echo "ERROR: uv is not executable at '${uv_bin}'" >&2
  exit 1
fi

mkdir -p "$(dirname "${uv_path_file}")"
printf "%s\n" "${uv_bin}" > "${uv_path_file}"
