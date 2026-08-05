"""Framework adapters for exporting learned arrays to PONNI Safetensors files.

Every framework import is intentionally lazy.  Installing PONNI does not pull
in a training framework merely to make the corresponding adapter importable.
Adapters preserve their documented parameter order and accept an optional
PONNI templated-model fingerprint.  When supplied, arrays are concatenated into
the single ``parameters`` tensor consumed by ``ponni::Inference::load_weights``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CompilerError
from .weights import write_ponni_file


def _validate_onnx_model(onnx_path: str | Path | None, framework: str,
                         validate_onnx: bool) -> dict[str, object] | None:
    """Validate generator compatibility and enrich failures with ONNX inventory."""
    if not validate_onnx:
        return None
    if onnx_path is None:
        raise CompilerError(
            f"{framework} generator export requires onnx_path so PONNI can verify that the model is supported; "
            "pass validate_onnx=False only for an intentional weight-only or templated-model export"
        )
    try:
        import onnx

        from .compiler import validate_model

        return validate_model(onnx_path)
    except ImportError as exc:
        raise CompilerError("ONNX compatibility validation requires the onnx package") from exc
    except CompilerError as exc:
        try:
            model = onnx.load(Path(onnx_path), load_external_data=False)
            opsets = {
                (entry.domain or "ai.onnx"): int(entry.version)
                for entry in model.opset_import
            }
            operations: dict[str, int] = {}
            node_details = []
            for index, node in enumerate(model.graph.node):
                domain = node.domain or "ai.onnx"
                qualified = f"{domain}::{node.op_type}"
                operations[qualified] = operations.get(qualified, 0) + 1
                node_details.append(
                    f"  [{index}] name={node.name or '<unnamed>'!r}, op={qualified}, "
                    f"inputs={list(node.input)!r}, outputs={list(node.output)!r}"
                )
            boundaries = []
            for kind, values in (("input", model.graph.input), ("output", model.graph.output)):
                for value in values:
                    tensor_type = value.type.tensor_type
                    shape = []
                    for dimension in tensor_type.shape.dim:
                        shape.append(
                            int(dimension.dim_value) if dimension.HasField("dim_value")
                            else dimension.dim_param or "?"
                        )
                    boundaries.append(
                        f"  {kind} {value.name!r}: elem_type={tensor_type.elem_type}, shape={shape}"
                    )
            detail_limit = 40
            details = node_details[:detail_limit]
            if len(node_details) > detail_limit:
                details.append(f"  ... {len(node_details) - detail_limit} additional nodes omitted")
            raise CompilerError(
                f"{framework} ONNX model is not supported by the PONNI generator.\n"
                f"PONNI diagnostic: {exc}\n"
                f"ONNX path: {onnx_path}\n"
                f"IR version: {model.ir_version}; opsets: {opsets}\n"
                f"Operator inventory: {dict(sorted(operations.items()))}\n"
                f"Boundaries:\n" + "\n".join(boundaries) + "\n"
                f"Nodes (up to {detail_limit}):\n" + "\n".join(details) + "\n"
                "Run `python -m kokkos_nn validate <model.onnx>` for the full PONNI analysis."
            ) from exc
        except CompilerError:
            raise
        except Exception as inventory_error:
            raise CompilerError(
                f"{framework} ONNX model is not supported by the PONNI generator. "
                f"PONNI diagnostic: {exc}. ONNX inventory also failed: {inventory_error}"
            ) from exc


def _numpy(value: Any) -> np.ndarray:
    """Move a framework tensor to the host without retaining framework state."""
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _write_arrays(arrays: Iterable[tuple[str, Any]], path: str | Path, *, source_framework: str,
                  model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                  validate_onnx: bool | None = None) -> dict[str, object]:
    should_validate = model_fingerprint is None if validate_onnx is None else validate_onnx
    onnx_report = _validate_onnx_model(onnx_path, source_framework, should_validate)
    normalized: list[tuple[str, np.ndarray]] = [(name, np.ascontiguousarray(_numpy(value))) for name, value in arrays]
    if not normalized:
        raise CompilerError(f"{source_framework} model does not expose any weight arrays")
    names = [name for name, _ in normalized]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise CompilerError(f"{source_framework} model exposes duplicate parameter names: {duplicates}")
    if model_fingerprint is not None:
        dtype = normalized[0][1].dtype
        if any(array.dtype != dtype for _, array in normalized):
            raise CompilerError("templated PONNI exports require all parameter arrays to use one dtype")
        parameters = np.concatenate([array.reshape(-1) for _, array in normalized])
        result = write_ponni_file(
            {"parameters": parameters}, path, model_fingerprint=model_fingerprint,
            source_framework=source_framework, target="template",
            metadata={"ponni.parameter_order": ",".join(name for name, _ in normalized)},
        )
    else:
        result = write_ponni_file(
            dict(normalized), path, source_framework=source_framework, target="generic",
        )
    if onnx_report is not None:
        result["onnx_validation"] = onnx_report
    return result


def export_keras_weights(model: Any, path: str | Path, *,
                         model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                         validate_onnx: bool | None = None) -> dict[str, object]:
    """Export ``keras.Model.weights`` in Keras' stable model traversal order."""
    arrays = []
    for index, variable in enumerate(model.weights):
        name = getattr(variable, "path", None) or getattr(variable, "name", None) or f"parameter.{index}"
        arrays.append((str(name).removesuffix(":0"), variable))
    return _write_arrays(arrays, path, source_framework="keras", model_fingerprint=model_fingerprint,
                         onnx_path=onnx_path, validate_onnx=validate_onnx)


def export_tensorflow_weights(module: Any, path: str | Path, *,
                              model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                              validate_onnx: bool | None = None) -> dict[str, object]:
    """Export a ``tf.Module`` or ``tf.keras.Model`` through its tracked variables."""
    variables = getattr(module, "weights", None)
    if variables is None:
        variables = getattr(module, "variables", None)
    if variables is None:
        raise CompilerError("TensorFlow object has neither weights nor variables")
    arrays = []
    for index, variable in enumerate(variables):
        name = str(getattr(variable, "name", f"parameter.{index}")).removesuffix(":0")
        arrays.append((name, variable))
    return _write_arrays(arrays, path, source_framework="tensorflow", model_fingerprint=model_fingerprint,
                         onnx_path=onnx_path, validate_onnx=validate_onnx)


def export_pytorch_weights(model: Any, path: str | Path, *, model_fingerprint: str | None = None,
                           transpose_linear_weights: bool = True, onnx_path: str | Path | None = None,
                           validate_onnx: bool | None = None) -> dict[str, object]:
    """Export a PyTorch state dict, canonicalizing two-dimensional Linear weights.

    PONNI Matvec stores weights as ``(input, output)`` while ``torch.nn.Linear``
    stores them as ``(output, input)``.  The default transpose is appropriate
    for PONNI's dense-network scope and can be disabled for an explicit mapping.
    """
    if not hasattr(model, "state_dict"):
        raise CompilerError("PyTorch object does not provide state_dict()")
    arrays = []
    for name, value in model.state_dict().items():
        array = _numpy(value)
        if transpose_linear_weights and name.endswith("weight") and array.ndim == 2:
            array = array.T
        arrays.append((str(name), array))
    return _write_arrays(arrays, path, source_framework="pytorch", model_fingerprint=model_fingerprint,
                         onnx_path=onnx_path, validate_onnx=validate_onnx)


def _flatten_tree(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_tree(value[key], child))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            result.extend(_flatten_tree(item, child))
        return result
    return [(prefix or "parameter", value)]


def export_jax_flax_weights(parameters: Any, path: str | Path, *,
                            model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                            validate_onnx: bool | None = None) -> dict[str, object]:
    """Export a JAX/Flax parameter PyTree using deterministic dictionary paths."""
    return _write_arrays(
        _flatten_tree(parameters), path, source_framework="jax-flax", model_fingerprint=model_fingerprint,
        onnx_path=onnx_path, validate_onnx=validate_onnx,
    )


def export_sklearn_weights(estimator: Any, path: str | Path, *,
                           model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                           validate_onnx: bool | None = None) -> dict[str, object]:
    """Export fitted scikit-learn MLP coefficients and intercepts layer by layer."""
    coefficients = getattr(estimator, "coefs_", None)
    intercepts = getattr(estimator, "intercepts_", None)
    if coefficients is None or intercepts is None or len(coefficients) != len(intercepts):
        raise CompilerError("scikit-learn exporter requires a fitted MLP estimator")
    arrays = []
    for index, (weights, bias) in enumerate(zip(coefficients, intercepts, strict=True)):
        arrays.extend(((f"layer.{index}.weight", weights), (f"layer.{index}.bias", bias)))
    return _write_arrays(arrays, path, source_framework="scikit-learn", model_fingerprint=model_fingerprint,
                         onnx_path=onnx_path, validate_onnx=validate_onnx)


def export_paddle_weights(model: Any, path: str | Path, *,
                          model_fingerprint: str | None = None, onnx_path: str | Path | None = None,
                          validate_onnx: bool | None = None) -> dict[str, object]:
    """Export PaddlePaddle named parameters; Paddle Linear is already input/output ordered."""
    if not hasattr(model, "named_parameters"):
        raise CompilerError("PaddlePaddle object does not provide named_parameters()")
    return _write_arrays(
        ((str(name), value) for name, value in model.named_parameters()),
        path, source_framework="paddlepaddle", model_fingerprint=model_fingerprint,
        onnx_path=onnx_path, validate_onnx=validate_onnx,
    )
