#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <uv_path_file> <uv_install_dir>" >&2
  exit 2
fi

uv_path_file="$1"
uv_install_dir="$2"
local_uv_bin="${uv_install_dir}/bin/uv"
export UV_CACHE_DIR="${uv_install_dir}/cache"
export UV_NO_CONFIG=1

mkdir -p "${uv_install_dir}/bin"

if [[ -x "${local_uv_bin}" ]]; then
  if ! "${local_uv_bin}" --version >/dev/null 2>&1; then
    echo "ERROR: the PONNI-owned uv at '${local_uv_bin}' is not valid on this machine" >&2
    echo "Remove or replace it only with explicit user permission" >&2
    exit 1
  fi
else
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | \
      env UV_UNMANAGED_INSTALL="${uv_install_dir}/bin" UV_NO_MODIFY_PATH=1 sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | \
      env UV_UNMANAGED_INSTALL="${uv_install_dir}/bin" UV_NO_MODIFY_PATH=1 sh
  else
    echo "ERROR: neither curl nor wget is available to install the PONNI-owned uv" >&2
    exit 1
  fi
fi

if [[ ! -x "${local_uv_bin}" ]] || ! "${local_uv_bin}" --version >/dev/null 2>&1; then
  echo "ERROR: PONNI-owned uv installation failed at '${local_uv_bin}'" >&2
  exit 1
fi

mkdir -p "$(dirname "${uv_path_file}")"
printf "%s\n" "${local_uv_bin}" > "${uv_path_file}"
