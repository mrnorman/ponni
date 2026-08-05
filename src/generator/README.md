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

Legal dense pairs can be streamed instead of materialized, and the two highest workspace-reduction levels can
recompute selected shared branches. The generator does not use architecture-specific thresholds, launch bounds, team
scratch, or an autotuner.

Nested static `Concat` and `Gather` results are composed into virtual dense-input index maps, and a `Gather` selecting
dense output rows prunes the unused weights and bias entries. Ordered dense epilogues, mixed predicate/selection
regions, linear and reconverging pointwise regions, and pointwise DAGs feeding one-pass reductions are fused without
reassociating arithmetic. Stable Softmax/LogSoftmax, LayerNormalization, and common decomposed activation spellings
are canonicalized before general region fusion. The local-storage planner applies reviewed in-place aliases and
compares deterministic arena placements; dense streaming and small recomputation sets are scored against planned
arena high-water.

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

Both commands print their JSON report by default. Add `--quiet` to suppress that terminal output;
`compile` still writes its report files, and the CMake generation targets use this mode.

Add `--onnx-preprocess` to run provider-neutral ONNX Script constant folding, function inlining, shape inference, and
dead-code cleanup before importing PONNI's canonical IR. This is opt-in so exporter compatibility remains explicit;
PONNI still validates the rewritten graph against its reviewed ONNX contract and compares the final optimized IR with
the original model.

Add `--analyze-workspace` to compare the selected native arena layout with the original greedy placement and an exact
small-graph oracle. PONNI exhaustively places up to seven live storage groups in normal compilation, which removes
arena fragmentation without changing generated arithmetic. The analysis mode extends exhaustive search through nine
groups. For larger layouts it uses deterministic OR-Tools CP-SAT when the optional `exact` dependency is installed:

```bash
python -m pip install -e 'src/generator[exact]'
python -m kokkos_nn validate model.onnx --analyze-workspace
```

The oracle covers arena placement for the already selected fusion, streaming, and recomputation schedule; it does not
claim global optimality over every possible graph rewrite or recomputation strategy.

### Workspace-reduction aggressiveness

`--workspace-reduction-aggressiveness 1-5` controls how aggressively PONNI avoids materialized intermediate
workspace. The default is level 3, which matches PONNI's standard deterministic dense-chain scheduling.

| Level | Dense streaming | Shared branches |
|---|---|---|
| 1 | Disabled | Always materialized |
| 2 | Only a legal dense pair whose consumer produces the model output | Always materialized |
| 3 | Legal non-overlapping pairs throughout the graph, maximizing total `H - O` | Always materialized |
| 4 | Level 3 search around selected branches | Recompute a two-consumer terminal dense branch only when planned workspace decreases |
| 5 | Level 3 search around selected branches | Recompute eligible one-hop dense branches of any fan-out when planned workspace decreases |

Here `H` is the eliminated hidden width and `O` is the consumer output width. Levels 4 and 5 extend the producer
input's planned lifetime through every recomputed consumer. Recursive recomputation is prohibited. Level 4 introduces
at most one additional producer evaluation; level 5 may duplicate more work for higher-fan-out branches.

For example:

```bash
python -m kokkos_nn compile model.onnx --output-dir generated \
  --workspace-reduction-aggressiveness 4
```

The optimization report records the selected level, streamed pairs, recomputed activations, eliminated workspace
elements, and additional dense multiply-adds. These levels describe generated workspace policy, not measured hardware
register allocation.

Disable an optimization pass when diagnosing a graph:

```bash
python -m kokkos_nn compile model.onnx --output-dir generated \
  --disable-pass dense-epilogue-fusion
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

`canonical_ir.json` is the canonical graph after optimization. `optimization_report.json` records original,
optimized, and nested fused-component operation lists, pass results, storage slots, dense streaming decisions, fusion
rejections, schema/opset metadata, and the three generated targets.

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
