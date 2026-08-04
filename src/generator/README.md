# PONNI Kokkos neural-network generator

PONNI compiles a constrained, fixed-feature ONNX inference graph into deterministic Kokkos C++ and a versioned
weight blob. Python and ONNX are build-time dependencies only. Generated inference depends on PONNI and Kokkos.

The complete operator matrix is generated in [ONNX_OPERATOR_SUPPORT.md](ONNX_OPERATOR_SUPPORT.md). It records the
reviewed ONNX schema range, supported restrictions, and unsupported operators.

## Pipeline

The compiler imports ONNX into PONNI's canonical graph, validates it, applies deterministic folding and fusion passes,
minimizes activation lifetimes, and assigns remaining floating-point and Boolean intermediates to reusable local
slots. It writes a generated header, `weights.bin`, `weights.json`, `canonical_ir.json`, and
`optimization_report.json`.

Legal dense pairs are streamed only when doing so reduces the number of live scalar values. Shared branch values are
materialized once and retained through their final consumer. The generator does not use architecture-specific
thresholds, recomputation cost rules, launch bounds, team scratch, or an autotuner.

## Installation and use

Install the build-time package in a Python environment containing NumPy and ONNX:

```bash
python -m pip install -e src/generator
```

Compile an exported model:

```bash
python -m kokkos_nn validate model.onnx
python -m kokkos_nn compile model.onnx --output-dir generated --model-name MyModel
```

Disable an optimization pass when diagnosing a graph:

```bash
python -m kokkos_nn compile model.onnx --output-dir generated \
  --disable-pass fuse_dense_bias_activation
```

`python -m kokkos_nn list-passes` prints the available deterministic passes.

## ONNX contract

Inputs and outputs use `(features, batch)` orientation. Feature dimensions must be static; the batch dimension may be
symbolic. Export helpers normalize framework orientation and attach PONNI metadata. The importer records ONNX IR,
domain opsets, resolved operator schema versions, and materialized default attributes in the optimization report.

The generated support table is authoritative for operator restrictions. In particular, reductions and normalization
operate over feature axes, and unsupported dynamic shape manipulation is rejected with a diagnostic.

Regenerate or check the table with:

```bash
PYTHONPATH=src/generator python src/generator/examples/generate_operator_support.py
PYTHONPATH=src/generator python src/generator/examples/generate_operator_support.py --check
```

## Canonical IR

The canonical graph is PONNI's internal, inference-oriented representation; it is not ONNX IR. It contains typed
tensors, constants, explicit producer/consumer links, canonical operations, normalized attributes, and original ONNX
compatibility metadata. Framework-specific transpose and reshape scaffolding is folded away when proven to preserve
the feature-major representation.

`canonical_ir.json` is the canonical graph after optimization. `optimization_report.json` records original and
optimized operation lists, pass results, storage slots, dense streaming decisions, fusion rejections, schema/opset
metadata, and the three generated targets.

## Generated APIs

Every generated model exposes exactly these inference APIs:

| API | Execution model | Arithmetic |
|---|---|---|
| `infer_one` | `KOKKOS_INLINE_FUNCTION`, caller-owned `SArray` input/output | model scalar type |
| `infer_batch` | one `Kokkos::RangePolicy` iteration per sample | model scalar type |
| `infer_batch_half2` | one `Kokkos::RangePolicy` iteration per adjacent sample pair | `ponni::TwoHalf` FP16 |

`infer_one` is intended for embedding in an existing device kernel. `infer_batch` owns a standalone Kokkos launch.
`infer_batch_half2` packs two adjacent samples, uses one dependent FP16 accumulation chain for each dense dot product,
and writes the valid lane when the batch size is odd. It has lower-precision semantics than the scalar APIs.

All three paths support the complete operator set described in `ONNX_OPERATOR_SUPPORT.md`. Boolean intermediates use
compact byte or `TwoMask` local storage. No generated inference API requests Kokkos team scratch.

## Weights and learned parameters

`weights.bin` has a fixed header with magic, version, scalar metadata, payload size, and checksum. `weights.json`
describes tensor offsets and learned parameters. `load_weights()` validates the blob and creates persistent scalar and
FP16 device views. `get_parameters()`, `set_parameters()`, `save_parameters()`, and `refresh_half_parameters()` support
online parameter updates while keeping both representations synchronized.

## Verification

Compilation runs the original and optimized canonical graphs on deterministic inputs and rejects an optimization if
their results disagree. Unit tests also compare exported models against CPU ONNX Runtime, compile generated headers,
and exercise all three inference APIs across PyTorch, Keras, TensorFlow, and operator-zoo examples.

The unit-test Python environment is unified and CPU-only because ONNX emission and reference semantics are independent
of the Kokkos backend. CUDA or HIP is needed only when generated C++ tests are configured for those backends.
