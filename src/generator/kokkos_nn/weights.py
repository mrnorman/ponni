"""Read and write the PONNI profile of the Safetensors container format.

The tensor table and packed payload remain ordinary Safetensors, so standard
Safetensors tools can inspect a ``.ponni`` file.  PONNI requires four metadata
entries in addition to the standard tensor descriptors: a profile version, a
model fingerprint, a tensor-schema fingerprint, and an FNV-1a payload checksum.
Generated C++ validates all four and every expected tensor before allocating
device storage.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Mapping

import numpy as np

from .errors import CompilerError
from .ir import DType, Graph


FORMAT_VERSION = 1
MAX_HEADER_BYTES = 100 * 1024 * 1024
METADATA_KEY = "__metadata__"
PROFILE_VERSION_KEY = "ponni.profile_version"
MODEL_FINGERPRINT_KEY = "ponni.model_fingerprint"
SCHEMA_FINGERPRINT_KEY = "ponni.schema_fingerprint"
PAYLOAD_CHECKSUM_KEY = "ponni.payload_checksum_fnv1a64"
SOURCE_FRAMEWORK_KEY = "ponni.source_framework"
TARGET_KEY = "ponni.target"

_NUMPY_TO_SAFETENSORS = {
    "bool": "BOOL",
    "int8": "I8",
    "uint8": "U8",
    "int16": "I16",
    "uint16": "U16",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
}


def fnv1a64(data: bytes) -> int:
    """Return the checksum also used by the header-only C++ reader."""
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _fingerprint_text(text: str) -> str:
    return f"fnv1a64:{fnv1a64(text.encode('utf-8')):016x}"


def _little_endian_contiguous(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.name not in _NUMPY_TO_SAFETENSORS:
        raise CompilerError(f"PONNI files do not support NumPy dtype {array.dtype}")
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    # np.ascontiguousarray promotes a rank-zero array to shape (1,), which
    # would make the file descriptor disagree with the generated scalar spec.
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def tensor_schema_fingerprint(tensors: Mapping[str, np.ndarray]) -> str:
    """Fingerprint names, Safetensors dtypes, and shapes in lexical name order."""
    lines = ["ponni-tensor-schema-v1"]
    for name in sorted(tensors):
        array = _little_endian_contiguous(tensors[name])
        dtype = _NUMPY_TO_SAFETENSORS[array.dtype.name]
        shape = ",".join(str(int(dimension)) for dimension in array.shape)
        lines.append(f"{name}\t{dtype}\t{shape}")
    return _fingerprint_text("\n".join(lines) + "\n")


def graph_fingerprint(graph: Graph) -> str:
    """Fingerprint the canonical graph schema without incorporating weight values."""
    graph_data = graph.to_dict()
    # Import/export provenance does not change which generated kernel accepts
    # the weights. Everything else describes the optimized tensor/operator ABI.
    graph_data["metadata"] = {}
    canonical = json.dumps(graph_data, sort_keys=True, separators=(",", ":"))
    return _fingerprint_text("ponni-generated-model-v1\n" + canonical)


def write_ponni_file(tensors: Mapping[str, np.ndarray], path: str | Path, *,
                     model_fingerprint: str | None = None,
                     source_framework: str = "ponni",
                     target: str = "generic",
                     metadata: Mapping[str, str] | None = None) -> dict[str, object]:
    """Write a standard Safetensors file with PONNI's required validation metadata."""
    if METADATA_KEY in tensors:
        raise CompilerError(f"{METADATA_KEY!r} is reserved for Safetensors metadata")

    normalized: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise CompilerError("PONNI tensor names must be non-empty strings")
        if name in normalized:
            raise CompilerError(f"duplicate PONNI tensor name {name!r}")
        normalized[name] = _little_endian_contiguous(value)

    ordered_names = sorted(normalized)
    payload_parts: list[bytes] = []
    descriptors: dict[str, dict[str, object]] = {}
    offset = 0
    for name in ordered_names:
        array = normalized[name]
        payload = array.tobytes(order="C")
        descriptors[name] = {
            "dtype": _NUMPY_TO_SAFETENSORS[array.dtype.name],
            "shape": [int(dimension) for dimension in array.shape],
            "data_offsets": [offset, offset + len(payload)],
        }
        payload_parts.append(payload)
        offset += len(payload)

    payload = b"".join(payload_parts)
    schema_fingerprint = tensor_schema_fingerprint(normalized)
    required_metadata = {
        PROFILE_VERSION_KEY: str(FORMAT_VERSION),
        MODEL_FINGERPRINT_KEY: model_fingerprint or schema_fingerprint,
        SCHEMA_FINGERPRINT_KEY: schema_fingerprint,
        PAYLOAD_CHECKSUM_KEY: f"fnv1a64:{fnv1a64(payload):016x}",
        SOURCE_FRAMEWORK_KEY: source_framework,
        TARGET_KEY: target,
    }
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CompilerError("Safetensors metadata keys and values must be strings")
        if key in required_metadata and required_metadata[key] != value:
            raise CompilerError(f"metadata cannot override required PONNI key {key!r}")
        required_metadata[key] = value

    header_object: dict[str, object] = {METADATA_KEY: required_metadata}
    header_object.update(descriptors)
    header = json.dumps(header_object, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    if len(header) > MAX_HEADER_BYTES:
        raise CompilerError(f"PONNI Safetensors header exceeds {MAX_HEADER_BYTES} bytes")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return {
        "format": "ponni-safetensors",
        "format_version": FORMAT_VERSION,
        "header_bytes": len(header),
        "payload_bytes": len(payload),
        "payload_checksum_fnv1a64": required_metadata[PAYLOAD_CHECKSUM_KEY],
        "model_fingerprint": required_metadata[MODEL_FINGERPRINT_KEY],
        "schema_fingerprint": schema_fingerprint,
        "tensor_count": len(descriptors),
        "tensors": [
            {"name": name, **descriptors[name], "byte_offset": 8 + len(header) + descriptors[name]["data_offsets"][0]}
            for name in ordered_names
        ],
    }


def _read_header(path: str | Path) -> tuple[bytes, dict[str, object], bytes]:
    blob = Path(path).read_bytes()
    if len(blob) < 10:
        raise CompilerError(f"PONNI file {path} is shorter than a Safetensors header")
    header_bytes = struct.unpack_from("<Q", blob)[0]
    if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES:
        raise CompilerError(f"PONNI file {path} has invalid Safetensors header size {header_bytes}")
    data_start = 8 + header_bytes
    if data_start > len(blob):
        raise CompilerError(f"PONNI file {path} ends inside its Safetensors header")
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CompilerError(f"PONNI file {path} has duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        header = json.loads(blob[8:data_start].decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompilerError(f"PONNI file {path} has invalid Safetensors JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CompilerError("Safetensors header root must be an object")
    return blob, header, blob[data_start:]


def validate_weight_blob(path: str | Path, manifest: dict[str, object] | None = None) -> dict[str, object]:
    """Validate the complete PONNI Safetensors profile and return its manifest."""
    _, header, payload = _read_header(path)
    metadata = header.get(METADATA_KEY)
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                 for k, v in metadata.items()):
        raise CompilerError("PONNI Safetensors metadata must be a string-to-string object")
    required = (PROFILE_VERSION_KEY, MODEL_FINGERPRINT_KEY, SCHEMA_FINGERPRINT_KEY, PAYLOAD_CHECKSUM_KEY)
    missing = [key for key in required if key not in metadata]
    if missing:
        raise CompilerError(f"PONNI Safetensors metadata is missing {', '.join(missing)}")
    if metadata[PROFILE_VERSION_KEY] != str(FORMAT_VERSION):
        raise CompilerError(f"unsupported PONNI profile version {metadata[PROFILE_VERSION_KEY]!r}")
    expected_checksum = f"fnv1a64:{fnv1a64(payload):016x}"
    if metadata[PAYLOAD_CHECKSUM_KEY] != expected_checksum:
        raise CompilerError(
            f"PONNI payload checksum mismatch: expected {metadata[PAYLOAD_CHECKSUM_KEY]}, got {expected_checksum}"
        )

    tensors: dict[str, np.ndarray] = {}
    ranges: list[tuple[int, int, str]] = []
    dtype_to_numpy = {value: key for key, value in _NUMPY_TO_SAFETENSORS.items()}
    for name, descriptor in header.items():
        if name == METADATA_KEY:
            continue
        if not isinstance(descriptor, dict):
            raise CompilerError(f"Safetensors descriptor for {name!r} must be an object")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in dtype_to_numpy or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise CompilerError(f"Safetensors descriptor for {name!r} is incomplete or unsupported")
        if not all(type(value) is int and value >= 0 for value in [*shape, *offsets]):
            raise CompilerError(f"Safetensors descriptor for {name!r} contains invalid dimensions or offsets")
        begin, end = offsets
        if begin > end or end > len(payload):
            raise CompilerError(f"Safetensors tensor {name!r} lies outside the payload")
        dtype_np = np.dtype(dtype_to_numpy[dtype]).newbyteorder("<")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - begin != elements * dtype_np.itemsize:
            raise CompilerError(f"Safetensors tensor {name!r} byte length does not match its dtype and shape")
        tensors[name] = np.frombuffer(payload[begin:end], dtype=dtype_np).reshape(shape)
        ranges.append((begin, end, name))

    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise CompilerError(f"Safetensors payload has a hole or overlap before tensor {name!r}")
        cursor = end
    if cursor != len(payload):
        raise CompilerError("Safetensors payload contains unindexed trailing bytes")
    actual_schema = tensor_schema_fingerprint(tensors)
    if metadata[SCHEMA_FINGERPRINT_KEY] != actual_schema:
        raise CompilerError("PONNI tensor-schema fingerprint does not match the Safetensors descriptors")
    result = {
        "format_version": FORMAT_VERSION,
        "payload_bytes": len(payload),
        "payload_checksum_fnv1a64": expected_checksum,
        "model_fingerprint": metadata[MODEL_FINGERPRINT_KEY],
        "schema_fingerprint": actual_schema,
        "tensor_count": len(tensors),
        "metadata": metadata,
        "tensors": [
            {"name": name, "dtype": header[name]["dtype"], "shape": header[name]["shape"],
             "data_offsets": header[name]["data_offsets"]}
            for name in sorted(tensors)
        ],
    }
    if manifest is not None:
        for key in (
            "format_version", "payload_bytes", "payload_checksum_fnv1a64",
            "model_fingerprint", "schema_fingerprint", "tensor_count",
        ):
            if manifest.get(key) != result[key]:
                raise CompilerError(f"weight manifest field {key!r} does not match the PONNI file")
    return result


def write_weights(graph: Graph, output_dir: Path) -> tuple[dict[int, int], dict[str, object]]:
    """Write canonical constants into ``weights.ponni`` in deterministic lexical-name order."""
    dtype = graph.tensors[graph.inputs[0]].dtype
    numpy_dtype = np.dtype("<f4" if dtype == DType.FLOAT32 else "<f8")
    tensors_by_id = [(tensor_id, tensor) for tensor_id, tensor in sorted(graph.tensors.items()) if tensor.is_constant]
    tensors: dict[str, np.ndarray] = {}
    tensor_ids_by_name: dict[str, int] = {}
    for tensor_id, tensor in tensors_by_id:
        if tensor.constant_name is None:
            raise CompilerError(f"constant tensor {tensor.name!r} has no constant payload")
        array = np.asarray(graph.constants[tensor.constant_name].values, dtype=numpy_dtype, order="C")
        tensors[tensor.name] = array
        tensor_ids_by_name[tensor.name] = tensor_id

    # Safetensors writers conventionally order the data buffer by tensor name.
    # The emitter's parameter offsets must follow that physical order rather
    # than the canonical IR's integer tensor IDs.
    offsets: dict[int, int] = {}
    element_offset = 0
    for name in sorted(tensors):
        offsets[tensor_ids_by_name[name]] = element_offset
        element_offset += tensors[name].size

    model_fingerprint = graph_fingerprint(graph)
    manifest = write_ponni_file(
        tensors, output_dir / "weights.ponni", model_fingerprint=model_fingerprint,
        source_framework="onnx", target="generated",
    )
    manifest["scalar_type"] = dtype.value
    manifest["model_num_inputs"] = graph.tensors[graph.inputs[0]].sample_size
    manifest["model_num_outputs"] = graph.tensors[graph.outputs[0]].sample_size
    manifest["learned_parameter_count"] = sum(
        graph.tensors[tensor_id].sample_size
        for tensor_id, _ in tensors_by_id
        if graph.constants[graph.tensors[tensor_id].constant_name].learned
    )
    constants_by_name = {
        tensor.name: graph.constants[tensor.constant_name]
        for _, tensor in tensors_by_id
    }
    for entry in manifest["tensors"]:
        constant = constants_by_name[entry["name"]]
        entry["canonical_layout"] = constant.canonical_layout
        entry["learned"] = constant.learned
        entry["element_offset"] = entry["data_offsets"][0] // numpy_dtype.itemsize
        entry["byte_size"] = entry["data_offsets"][1] - entry["data_offsets"][0]
    (output_dir / "weights.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return offsets, manifest
