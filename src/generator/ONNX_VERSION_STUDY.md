# CPU ONNX exporter version study

Date: 2026-08-03

This study ran the existing PONNI PyTorch, Keras, and TensorFlow example models through three isolated CPU-only
framework stacks. The reproducible driver is `examples/probe_onnx_versions.py`, and the exact package definitions are
in `examples/onnx_version_matrix.json`.

## Tested stacks

| Stack | PyTorch | Keras | TensorFlow | tf2onnx | ONNX | ONNX Runtime | ONNX Script | ONNX IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early | 2.6.0+cpu | 3.8.0 | 2.18.1 CPU | 1.16.1 | 1.17.0 | 1.20.1 | 0.2.2 | n/a |
| Middle | 2.9.1+cpu | 3.11.3 | 2.20.0 CPU | 1.17.0 | 1.19.1 | 1.23.2 | 0.5.6 | 0.1.11 |
| Current | 2.13.0+cpu | 3.15.1 | 2.21.0 CPU | 1.17.0 | 1.22.0 | 1.28.0 | 0.7.1 | 0.2.1 |

Keras used its TensorFlow backend. Every environment reported only a TensorFlow CPU device and a `+cpu` PyTorch
build.

## Results

- All 27 exports (nine models in each stack) succeeded and matched ONNX Runtime. The maximum absolute difference was
  `1.4901161193847656e-07`; the largest relative difference was `4.3855801777681336e-05` near zero.
- Keras emitted identical ONNX structure in all three stacks for both tested models. The MLP used
  `Transpose -> MatMul -> Add -> activation`, while normalization remained the same 20-node decomposition. All used
  ONNX opset 15 and IR version 8.
- The explicit TensorFlow/tf2onnx residual model emitted identical nine-node ONNX in all stacks, using opset 18 and IR
  version 8.
- PyTorch emitted identical operator sequences and initializer layouts for every model. PyTorch 2.6 used opset 18 and
  also declared `pkg.onnxscript.torch_lib.common:1`; PyTorch 2.9 and 2.13 used only `ai.onnx` opset 20. All used IR
  version 10.
- PyTorch 2.6 named the requested dynamic dimension `s0`, while PONNI's exporter metadata declared it as `batch`.
  Consequently, PONNI rejected all six otherwise-valid PyTorch 2.6 models at the boundary-shape check. A diagnostic
  retry changing only `ponni.batch_symbol` to `s0` accepted every model.
- After that metadata-only correction, each model had one canonical-operation variant, one optimized-operation
  variant, and one storage/scheduling variant across all three stacks. In particular, whole-graph fusion and temporary
  planning were unchanged.

PONNI now normalizes the actual exported PyTorch input/output batch dimension, as the Keras/TensorFlow annotation
helper already does, rather than assuming the exporter preserves the requested symbolic name.

The importer hardening prompted by this study is deliberately independent of particular emitters. It preserves
domain-specific opsets, resolves and records exact ONNX operator schema versions, materializes defaults, normalizes
equivalent optional-input spellings, and rejects unreviewed IR, opset, schema, dtype, and semantic combinations.
Direct schema fixtures complement this exporter matrix: the standard defines legal meanings, while the matrix detects
which legal decompositions and dependency interactions real framework versions choose.

## Dependency provenance finding

An initially unconstrained middle stack resolved ONNX IR 0.2.1 with ONNX Script 0.5.6. PyTorch 2.9 then failed in its
post-export ONNX version-conversion pass because model-local functions remained after its inline pass. Pinning the
contemporaneous ONNX IR 0.1.11 made all PyTorch 2.9 exports succeed. This did not expose a PONNI importer issue, but it
demonstrates that a reproducible exporter stack must pin ONNX Script and ONNX IR alongside the framework, ONNX, and
ONNX Runtime.

## Reproduction

```bash
python3 src/generator/examples/probe_onnx_versions.py \
  --output-dir /tmp/ponni-onnx-version-study
```

The output contains a provenance record (including the Git revision, matrix, command, and probe hash), full resolved
package lists, logs, exported models, numerical reports, raw ONNX summaries, PONNI analyses, compatibility retries,
and `comparison.json`.
