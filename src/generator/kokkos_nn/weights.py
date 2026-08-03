from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

import numpy as np

from .errors import CompilerError
from .ir import DType, Graph


MAGIC = b"PNNWGT1\0"
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sIIQQ")


def fnv1a64(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def write_weights(graph: Graph, output_dir: Path) -> tuple[dict[int, int], dict[str, object]]:
    dtype = graph.tensors[graph.inputs[0]].dtype
    numpy_dtype = np.dtype("<f4" if dtype == DType.FLOAT32 else "<f8")
    scalar_code = 1 if dtype == DType.FLOAT32 else 2
    tensors = [tensor for _, tensor in sorted(graph.tensors.items()) if tensor.is_constant]
    offsets: dict[int, int] = {}
    payload_parts: list[bytes] = []
    entries: list[dict[str, object]] = []
    element_offset = 0
    for tensor in tensors:
        if tensor.constant_name is None:
            raise CompilerError(f"constant tensor {tensor.name!r} has no constant payload")
        constant = graph.constants[tensor.constant_name]
        array = np.asarray(constant.values, dtype=numpy_dtype, order="C")
        payload = array.tobytes(order="C")
        offsets[tensor.id] = element_offset
        entries.append(
            {
                "tensor_id": tensor.id,
                "name": tensor.name,
                "shape": list(array.shape),
                "byte_offset": HEADER.size + element_offset * numpy_dtype.itemsize,
                "byte_size": len(payload),
                "element_offset": element_offset,
                "canonical_layout": constant.canonical_layout,
            }
        )
        payload_parts.append(payload)
        element_offset += array.size
    payload = b"".join(payload_parts)
    checksum = fnv1a64(payload)
    blob = HEADER.pack(MAGIC, FORMAT_VERSION, scalar_code, len(payload), checksum) + payload
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights.bin").write_bytes(blob)
    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "scalar_type": dtype.value,
        "endianness": "little",
        "header_bytes": HEADER.size,
        "payload_bytes": len(payload),
        "payload_checksum_fnv1a64": f"0x{checksum:016x}",
        "model_num_inputs": graph.tensors[graph.inputs[0]].sample_size,
        "model_num_outputs": graph.tensors[graph.outputs[0]].sample_size,
        "tensor_count": len(entries),
        "tensors": entries,
    }
    (output_dir / "weights.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return offsets, manifest


def validate_weight_blob(path: str | Path, manifest: dict[str, object] | None = None) -> dict[str, int]:
    blob = Path(path).read_bytes()
    if len(blob) < HEADER.size:
        raise CompilerError(f"weight file {path} is shorter than the {HEADER.size}-byte header")
    magic, version, scalar_code, payload_bytes, expected_checksum = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise CompilerError(f"weight file {path} has invalid magic {magic!r}")
    if version != FORMAT_VERSION:
        raise CompilerError(f"weight file {path} uses unsupported format version {version}")
    if scalar_code not in (1, 2):
        raise CompilerError(f"weight file {path} uses invalid scalar code {scalar_code}")
    if len(blob) != HEADER.size + payload_bytes:
        raise CompilerError(
            f"weight file {path} size mismatch: header declares {payload_bytes} payload bytes, "
            f"file contains {len(blob) - HEADER.size}"
        )
    actual_checksum = fnv1a64(blob[HEADER.size:])
    if actual_checksum != expected_checksum:
        raise CompilerError(
            f"weight file {path} checksum mismatch: expected 0x{expected_checksum:016x}, "
            f"got 0x{actual_checksum:016x}"
        )
    if manifest is not None:
        if manifest.get("format_version") != version or manifest.get("payload_bytes") != payload_bytes:
            raise CompilerError("weight manifest does not match binary header metadata")
        expected_endianness = "little" if sys.byteorder == "little" else "big"
        if manifest.get("endianness") != expected_endianness:
            raise CompilerError(f"weight manifest endianness {manifest.get('endianness')!r} is unsupported")
    return {"version": version, "scalar_code": scalar_code, "payload_bytes": payload_bytes,
            "checksum": expected_checksum}
