# PONNI Kokkos neural-network generator

This directory contains a working ahead-of-time compiler for fixed-shape, inference-only, floating-point MLP DAGs.
ONNX is an interchange format only: importing, validation, optimization, and code generation happen in Python, while
the generated inference code links only Kokkos and PONNI. Neither Python nor an ML runtime is present in the C++
inference executable.

See [PERFORMANCE.md](PERFORMANCE.md) for the current optimized single-precision GPU baseline and next experiments.

## Architecture

```text
PyTorch nn.Module / Keras Model / TensorFlow Module
  -> torch.onnx.export / Keras ONNX export / tf2onnx
  -> checked and shape-inferred ONNX
  -> framework-neutral per-sample IR
  -> deterministic canonicalization and fusion passes
  -> deterministic dense-chain schedule, liveness intervals, and reusable local storage
  -> generated Kokkos header + versioned binary weights + JSON reports
```

The components are deliberately separate: `importer.py` owns ONNX semantics, `ir.py` defines the serializable IR,
`passes.py` owns transformations, `scheduler.py` chooses dense-chain materialization/streaming/recomputation,
`planner.py` performs liveness/storage planning, `emitter.py` writes C++, and
`weights.py` owns the external format. `interpreter.py` provides an independent numerical check of unfused and fused
IR. No device-side graph interpreter or virtual dispatch is generated.

## Capability summary

The compiler targets feature-vector inference DAGs, not arbitrary framework programs. PyTorch users supply an
evaluated `torch.nn.Module` and its input width to `kokkos_nn.export.export_module()`. Keras 3 models can export ONNX
directly, and TensorFlow modules can use tf2onnx; `kokkos_nn.framework_export` contains tested reference exporters for
both paths. Each exporter makes the public `(features,batch)` boundary explicit even when framework-native dense
layers use `(batch,features)`. ONNX is then validated, converted to a per-sample IR, optimized, statically scheduled,
and emitted as C++; PyTorch, Keras, TensorFlow, ONNX, ONNX Runtime, and Python are absent from the inference
executable.

Currently useful network families include:

- dense MLPs of arbitrary practical depth, with equal or varying layer widths;
- residual networks whose skip paths use supported elementwise operations;
- branched and gated DAGs, including values with multiple consumers;
- DenseNet-like feature concatenation along the static feature axis;
- feature-vector pipelines using supported activations, normalization, softmax/log-softmax, reductions, and scalar
  or exact-shape elementwise arithmetic;
- float32 models, plus float64 ONNX models on the full-precision portable paths;
- a dynamic batch size with every non-batch dimension fixed at generation time.

This means familiar combinations of `nn.Linear`, activation modules/functions, residual addition, feature-axis
`torch.cat`, evaluation-mode BatchNorm, LayerNorm over the complete feature vector, and feature-axis probabilities or
reductions can compile when their exported ONNX spelling uses the supported operators below. Support is determined by
the validated ONNX graph, not merely by the PyTorch class name.

### ONNX compatibility contract

The importer accepts ONNX IR versions 8 through 13 and the standard `ai.onnx` opsets 13 through 22. It records every
domain-specific opset import and every resolved `Operator:since_version` schema in `optimization_report.json`.
Operator support is schema-versioned: encountering a familiar operator name with a newly revised ONNX schema is an
error until that schema has been reviewed. Custom-domain nodes are rejected; an unused custom-domain opset declaration
is retained as provenance.

Import performs ONNX checking and strict shape inference, validates node arity and attributes against the selected
standard schema, materializes schema defaults, and then lowers emitter variation to one internal spelling. In
particular, positional `Clip` bounds and reduction axes become canonical attributes, omitted `Transpose` permutations
become explicit, and older reduction/reshape defaults become explicit. Semantic restrictions narrower than ONNX—such
as full feature-axis reductions, `Reshape allowzero=0`, and inference-mode normalization—are rejected before
optimization or code generation. The framework exporters also rewrite actual input and output dimensions to the
declared `(features,batch)` contract, so exporter-chosen names such as `s0` cannot conflict with metadata.

This is intentionally a two-layer defense: the ONNX standard supplies the schema contract, while direct fixtures and
the CPU exporter matrix cover legal but emitter-dependent graph decompositions. See
[`ONNX_VERSION_STUDY.md`](ONNX_VERSION_STUDY.md) for the current matrix and provenance procedure.

The following model families are not currently supported:

- convolution, pooling, images with preserved spatial semantics, or general multidimensional tensor kernels;
- recurrent networks, loops, conditionals, stateful models, or runtime graph control flow;
- attention/Transformer graphs, embeddings, gather/scatter, or a runtime-varying sequence length;
- dynamic feature, hidden, output, spatial, or sequence dimensions;
- multiple model inputs or outputs, tuples, sequences, strings, sparse tensors, and custom ONNX domains;
- training-mode behavior, gradients, random operations, or a `Dropout` node that remains after export;
- quantized integer inference, arbitrary mixed precision, nonconstant dense weights, or runtime weight updates;
- arbitrary NumPy-style broadcasting, concatenation/reduction over batch, and reshapes that change batch-relative
  element order;
- convolutional, recurrent, or general-purpose ONNX execution through a generated runtime interpreter.

Each sample is flattened conceptually to one statically sized feature vector. Static reshapes and flattening are
accepted only when they preserve the number and order of per-sample elements and keep batch on the same side; an
explicit batch-axis `Transpose` is handled at the framework boundary. Input and output objects must not alias because
in-place inference is not part of the generated contract.

## Installation and end-to-end example

The normal repository build prepares these dependencies in `unit/build/python_env`. For a standalone environment:

```bash
cd src/generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD

python examples/export_models.py --output-dir /tmp/ponni_models
python examples/export_framework_models.py --output-dir /tmp/ponni_models
python -m kokkos_nn validate /tmp/ponni_models/mlp.onnx
python -m kokkos_nn compile /tmp/ponni_models/mlp.onnx \
  --output-dir /tmp/ponni_models/mlp_generated \
  --strategy auto --model-name MlpModel
```

The repository-integrated demonstration is reproducible with:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_debug.env
./cmakescript.sh
make -j generator_integration generator_benchmark
ctest -V -R generator
./generator/generator_benchmark generator/generated/mlp_generated/weights.bin
```

Use `thatchroof_gpu_debug.env` for correctness and device-memory diagnostics. Collect optimization walltimes in
single precision with `thatchroof_gpu_fast.env`:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_fast.env
./cmakescript.sh
make -j generator_gpu_scale generator_benchmark
ctest -V -R generator_gpu_scale_test
```

The CUDA-only scale test uses `I -> I -> I -> 3` networks for `I = 4, 8, 16, 32, 64, 128` and batches 10,000,
100,000, and 1,000,000. It reports all batched strategies, checks their outputs, and uses a 1 GiB PONNI device pool.
It is not registered when Kokkos CUDA is disabled.

The PyTorch example exporter creates `Linear -> Tanh -> Linear`, a shallow residual MLP, a varying-width depth-10 MLP, a
ten-dense/five-block ResNet, a concatenative DenseNet, a shared-trunk two-branch DAG that exercises bounded
recomputation, and an ONNX operator zoo. The framework exporter additionally creates a Keras dense MLP, a Keras
BatchNorm/LayerNorm/Softmax pipeline, and a pure TensorFlow residual MLP. It verifies each framework result against
ONNX Runtime and passes all three through the same importer, fusion, storage-planning, and C++ generation path. The compact
functionality models
use batches 1, 2, 3, 7, and 11; the original examples retain batches 1, 2, 7, 32, and 67. Export compares PyTorch and
ONNX Runtime and writes reproducible references. C++ integration checks SArray, direct View batch, hierarchical tile
1 and default tile, and packed half2 inference.

Focused framework tests target exporter representation differences rather than reproducing every PyTorch topology.
They cover Keras Functional branching, no-bias dense layers, feature concatenation, residual activation, decomposed
normalization, and Boolean-select ELU; TensorFlow `transpose_b` constant weights, `tf.nn.bias_add`, compile-time
reshape, shared branches, and actionable rejection of an exported unsupported op. Keras LayerNormalization currently
exports `Sqrt -> Reciprocal` plus reductions and elementwise operations, all of which are compiled directly. Keras
3's ELU export adds Boolean `Greater/Not/Cast` selection around `Elu`; the typed-mask path compiles this spelling.

To investigate exporter drift across CPU-only framework versions, run:

```bash
python3 src/generator/examples/probe_onnx_versions.py --output-dir /tmp/ponni-onnx-versions
```

The version stacks are defined in `examples/onnx_version_matrix.json`. The probe creates isolated `uv` environments,
exports the existing PyTorch, Keras, and TensorFlow examples, records exact package provenance and exporter logs, and
compares raw ONNX structure with PONNI's canonical IR, optimized operations, storage plans, and numerical checks.
Additional coherent stacks can be appended to the matrix without changing the probe.
The results from the initial three-stack CPU investigation are recorded in `ONNX_VERSION_STUDY.md`.

## Supported ONNX scope

The generated [`ONNX_OPERATOR_SUPPORT.md`](ONNX_OPERATOR_SUPPORT.md) table gives a complete operator-by-operator and
schema-version accounting for the supported opset envelope, including PONNI-specific semantic restrictions.

Supported standard-domain operations are:

- dense/layout: `Gemm`, constant-right-hand-side `MatMul`, static feature-axis `Concat` and constant-index `Gather`,
  `Identity`, order-preserving static `Reshape`/`Flatten`/`Squeeze`/`Unsqueeze`, and batch-axis or constant-weight
  `Transpose`; statically resolvable `Shape` and `Size` expressions are folded during import;
- arithmetic: binary `Add`, `Mul`, `Sub`, `Div`, `Min`, `Max`, and `Pow`; scalar-bound `Clip`; and unary `Abs`,
  `Neg`, `Exp`, `Log`, `Sqrt`, `Reciprocal`, `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`,
  `Acosh`, `Atanh`, `Erf`, `Ceil`, `Floor`, `Round`, and `Sign`; variadic `Mean` and `Sum`;
- Boolean/select: `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `And`, `Or`, `Xor`, `Not`, `Where`,
  `IsNaN`, `IsInf`, Boolean-to-floating `Cast`, and narrowly equivalent `CastLike`;
- activations: `Tanh`, `Relu`, `Sigmoid`, `LeakyRelu`, `Elu`, `Gelu`, `Softplus`, `HardSigmoid`, `HardSwish`, and
  `Mish`, plus `PRelu`, `Selu`, `Celu`, `Softsign`, and `ThresholdedRelu`; `Sigmoid(x) -> Mul(x, ...)` is
  canonicalized to scalar `Silu`;
- inference-mode `Dropout` elimination and `BatchNormalization`, feature-axis `LayerNormalization` and
  `LpNormalization`, and stable `Softmax` and `LogSoftmax`;
- `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceProd`,
  `ReduceSum`, and `ReduceSumSquare` over the complete static feature axis, with either `keepdims` setting.

Concatenating or reducing the dynamic batch axis is rejected. Every reduction is local to one sample, so generated
code uses an in-kernel scalar or `TwoHalf` accumulator without atomics or an additional launch. Inference `Dropout`
is removed; training-mode Dropout is rejected. Normalization parameters and Clip bounds must be compile-time
constants, and BatchNormalization training mode is rejected. `Gemm` attributes
`transA`, `transB`, `alpha`, and `beta` are interpreted explicitly; unsupported `transA` data layouts are rejected.
Weights are normalized to canonical `(num_outputs, num_inputs)` order before serialization, so inference never
transposes weights.

The packed `TwoHalf` path evaluates transcendental operations independently in two FP32 lanes through Kokkos device
math and rounds each result back to FP16. It preserves ONNX operator semantics, including ties-to-even `Round` and
NaN-preserving `Sign`; FP16 quantization and backend math approximations can still differ numerically from ONNX
Runtime.

The importer accepts the reviewed IR/opset envelope stated above rather than trusting whatever version happens to be
installed. It accepts only nodes in the standard `ai.onnx` domain. Operators are checked by resolved schema and
semantics, not by framework-assigned names.

Exactly one floating-point input and output are currently supported. Both are logically rank two:

```text
input  = (num_inputs,  batch_size)
output = (num_outputs, batch_size)
```

The tested framework exporters use explicit boundary transposes around batch-major dense conventions and attach ONNX
metadata describing the feature-major boundary. Because Keras and tf2onnx may invent exporter-specific symbolic
dimension names, the boundary annotation normalizes only the declared input/output feature count and batch symbol;
ONNX validation and independent static-shape checks still validate the graph. The batch dimension must be symbolic;
every non-batch dimension must be positive and static. The canonical IR removes the batch dimension and describes one
sample. Moving only the batch axis, flattening, and same-element-count reshapes disappear at compile time.

The compiler rejects loops, graph-level conditions, sequences, strings, sparse tensors, random/stateful/training
operations, custom domains, runtime-dependent shape values, nonconstant dense weights, multiple dynamic dimensions,
and arbitrary broadcasting. Shape expressions are accepted only when their selected dimensions are independent of
runtime batch size. Elementwise broadcasting is limited to exact per-sample shapes and scalar constants. Layout
operations cannot move batch from first to last or place it between static dimensions because that would change
sample grouping rather than merely change metadata. Diagnostics name the offending node, operator, shape, or
attribute whenever available.

## Canonical IR and optimization

`TensorValue`, `Node`, `Graph`, and `ConstantTensor` form a small typed dataclass IR independent of ONNX protobufs.
Floating tensors retain their source precision and Boolean tensors use the distinct `bool` dtype; a JSON snapshot is
emitted as `canonical_ir.json`. The fixed deterministic pass order is:

1. topological scheduling;
2. constant folding;
3. identity elimination;
4. static layout-operation folding;
5. dead-node/tensor removal;
6. `Gemm` and `MatMul` dense canonicalization;
7. dense+bias fusion;
8. sigmoid-multiply to SiLU fusion;
9. dense+bias+activation fusion;
10. residual-add+activation fusion;
11. sole-consumer comparison-to-`Where` fusion;
12. elementwise-chain fusion;
13. dead-code cleanup and final scheduling.

Use `python -m kokkos_nn list-passes` for exact names and `--disable-pass NAME[,NAME]` to test a pass independently.
Fusion never silently crosses a multiply-consumed value. Such values are retained unless the explicit bounded-cost
recomputation rule is legal, and the report records the decision and cost.
`DenseBiasActivation` emits one accumulator scalar per output neuron; there is no preactivation array. Elementwise
chains reassign one scalar so a step such as `Pow` is not recomputed by later `Min`/`Max` operations. Softmax stores
shifted/exponential values in its planned output slot before scalar normalization. LayerNorm retains only scalar
Welford statistics, avoiding a tensor temporary and cancellation in `E[x^2] - E[x]^2`.

The storage planner computes producer/last-consumer intervals and reuses non-overlapping offsets. Floating
intermediates occupy the existing scalar workspace; Boolean intermediates occupy a separate byte-per-element mask
workspace, so adding masks does not inflate floating storage or disturb its reuse. Model inputs and outputs are
external, constants reside in the persistent weight view, and fused temporaries have no slot. A comparison used only
as a `Where` condition becomes canonical `CompareSelect`, eliminating even that mask allocation.
`optimization_report.json` lists both storage plans and their byte totals along with node counts, operations, fusion
rejections, external workspace bytes, and per-dense half2 accumulator selections. `--max-stack-bytes` rejects a model
whose combined estimate exceeds the explicit threshold. Local arrays may spill on GPUs; the report does not claim
register residency.

The deterministic dense-chain scheduler considers every legal `DenseBiasActivation -> Dense` edge whose consumer has
at most `--streaming-output-threshold` scalar accumulators (8 by default). On each linear chain it uses a
maximum-saved-elements non-overlapping matching: selected activations are produced one scalar at a time and streamed
into fixed output accumulators, while intervening activations are materialized. Multiple-consumer activations are
retained by default. A branch may instead be recomputed only when every consumer is a terminal eligible dense node
and the duplicated producer work is at most `--streaming-recompute-threshold` multiply-adds (64 by default); setting
that threshold to zero disables the optimization. This deliberately prevents recursive or exponential recomputation.

The sample-local storage plan excludes streamed values, accounts for delayed source uses introduced by recomputation,
and reports every activation decision and its reason in `dense_chain_schedule`. `infer_batch` stages every input
feature once in a fixed `SArray` before executing the same sample-local arithmetic body, avoiding repeated global View
loads. The report separates input staging, compact sample-local workspace, and the unchanged hierarchical scratch
plan. Hierarchical emission currently materializes the full graph because its neuron-parallel synchronization model
differs from scalar streaming.

## Generated APIs and scheduling

Every generated class exposes four inference families. The half2 family exposes its baseline API; an explicit API is
added when `--half2-accumulators` is supplied:

| # | Family | Generated API | Arithmetic and launch | Intended use |
|---|---|---|---|---|
| 1 | Inline SArray | `infer_one` | Full-precision `Scalar`; no internal launch | Embed one inference inside an existing Kokkos device kernel |
| 2 | View batch | `infer_batch` | Full-precision `Scalar`; one `RangePolicy` iteration per sample | Small MLPs and abundant batch parallelism |
| 3 | Hierarchical | `infer_batch_hierarchical` | Full-precision `Scalar`; `TeamPolicy`, neuron/batch work, planned team scratch | Graphs where neuron parallelism can offset team overhead |
| 4 | Packed two-sample half | `infer_batch_half2*` | FP16 weights/products/partials in two adjacent batch lanes; Kokkos `RangePolicy` | CUDA/HIP throughput when approximate FP16 semantics are acceptable |

The first three are the full-precision portable choices. The half2 family deliberately changes floating-point
semantics and is never selected by `auto`. `--strategy` records or enforces the recommended batched family; it does
not remove the other generated APIs.

```cpp
using InputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;
using OutputView = Kokkos::View<Scalar**,Kokkos::LayoutRight,ponni::DeviceSpace>;

template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
void infer_batch(InputView const & inputs, OutputView const & outputs) const;

template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
void infer_batch_hierarchical(InputView const & inputs, OutputView const & outputs,
                              int batch_tile = default_hierarchical_batch_tile) const;

template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
void infer_batch_half2(InputView const & inputs, OutputView const & outputs) const;

// Emitted only when requested at compile time.
template <unsigned MaxThreads = 0, unsigned MinBlocks = 0>
void infer_batch_half2_explicit(InputView const & inputs, OutputView const & outputs) const;

KOKKOS_INLINE_FUNCTION
void infer_one(ponni::SArray<Scalar,num_inputs> const & inputs,
               ponni::SArray<Scalar,num_outputs> & outputs) const;
```

`infer_one` contains fixed loops, scalars, and at most one fixed local workspace; eligible dense-chain pairs use only
scalar hidden/output temporaries, with non-overlapping pairs selected across deeper chains. It has no
batch index, allocation, exception,
virtual call, runtime graph traversal, or host-only object and is callable inside an existing Kokkos lambda.

Both batched targets validate extents in debug builds and launch Kokkos internally. `infer_batch` emits direct
View-based sample-local computation per `RangePolicy` iteration and spells out
`linear = iwork * batch_size + ibatch` through the equivalent
modulo/division mapping. Thus `ibatch` is fastest and accesses `view(feature,ibatch)` are contiguous under
`LayoutRight`.

`infer_batch_hierarchical` assigns one `TeamPolicy` team to a tile of samples and distributes each operation's
neuron-by-sample product with `TeamThreadRange`. Its flattened mapping is
`linear = neuron * active_batch + local_batch`, so batch remains fastest. Compiler-planned live activations use the
same batch-fastest layout in per-team scratch, with an explicit barrier between dependent graph operations. Each dense
neuron/sample pair has its own scalar accumulator and serial input reduction, so arithmetic order matches the other
targets. The runtime tile parameter permits device-specific measurement without producing another kernel
specialization; the emitted default is 32 for the measured `I = 4` through `128` suite, capped by
`--max-team-scratch-bytes` (48 KiB by default).
Every call checks the tile against the generated maximum before launching. The kernel captures only views and
scalars—never `this` or host-only state.

The `MaxThreads` and `MinBlocks` template parameters on every launched API map directly to
`Kokkos::LaunchBounds`. Their `0, 0` defaults preserve Kokkos's portable default behavior. Hierarchical batch tile
remains a runtime argument, so changing it does not instantiate another kernel.

The `--strategy` option records the recommended batched target; all four targets are generated. `auto`
prefers sample-local execution when the dense-chain scheduler selects streaming or bounded recomputation. Otherwise it recommends hierarchical execution
when an operation has at least `--team-output-threshold` output neurons (64 by default). The emitted weight view does
not currently use `Kokkos::RandomAccess`: short, regular,
output-major dense traversal has not demonstrated a benefit that justifies enabling it.

The half2 family is the fourth, Kokkos-launched target. A `RangePolicy` iteration processes two adjacent batch
samples as the lanes of `ponni::TwoHalf`; CUDA maps this type to `__half2`, HIP maps it to the corresponding
`__half2`, and other Kokkos backends use a correctness-oriented two-lane fallback. Inputs and outputs retain the
generated float/double View API. `load_weights()` creates one additional
persistent scalar-FP16 weight View, and each weight is splatted into both batch lanes in the dense loop—weights are
not duplicated as half2 values. Dense operations use packed FP16 multiply-add with FP16 accumulation, while
activations unpack to float for the Kokkos math function and repack afterward. `infer_batch_half2` uses one dependent
accumulation chain; this policy is reported as accumulator count 0.

Boolean lanes use the one-byte `ponni::TwoMask`: bit 0 corresponds to the low sample and bit 1 to the high sample.
Comparisons and logical operations stay packed, `Where` selects the two FP16 lanes independently, and
Boolean-to-floating `Cast` produces packed `0.0`/`1.0` lanes.

The generator retains a power-user heuristic that
selects a count independently for every dense dot product: 0 below length 2, 2 through length 24, 4 through length
80, and 16 above length 80, but it no longer emits a public heuristic method. A nonzero count creates that many
independent FP16 FMA chains. Their low and high lanes, plus the bias,
are converted to FP32 scalars and summed at the neuron boundary before repacking to FP16. This reduces dependency
length and usually reduces accumulation error, but does not make the dot product fully FP32. These thresholds come
from the documented Ampere `I -> I -> I -> 3` measurements and
are deterministic rather than runtime autotuning. A streamed consumer also caps its count so
`output_size * accumulators <= 48`, matching the largest measured live-output-partial point instead of extrapolating
its register demand to wider outputs.

`--half2-accumulators 0` explicitly requests the single-chain baseline. `--half2-accumulators 4` emits
`infer_batch_half2_explicit` with four partials for every canonical dense node.
A comma-separated list such as `--half2-accumulators 2,16,4` assigns counts in the dense order printed in the
optimization report. Supported explicit values are 0, 2, 4, 8, 16, and 32. No policy branch occurs during inference;
each API contains its selected straight-line reductions. `--strategy half2` recommends the baseline API, while
`auto` does not select approximate half2 semantics. All half2 APIs perform no inference-time allocation and handle
odd batch sizes. The one-sample `infer_one` contract cannot use two independent batch lanes, so a useful
SArray packed variant would require a separate two-sample API rather than changing `infer_one`.

## Generated launch autotuner

Compilation also emits `<Model>_autotune.cpp`. This standalone program instantiates a broad set of launch-bound
combinations for each launched inference family and combines them with every generated power-of-two hierarchical
batch tile. It fills deterministic random input data, performs three warmup runs, records nine fenced runs per
configuration, and uses the median. The warmup and timed-run constants are intentionally near the top of the source
so users can edit them easily.

The first command-line argument selects batch size and defaults to `1000000`; the optional second argument selects the
weight file and defaults to `weights.bin`:

```bash
./MlpModel_autotune
./MlpModel_autotune 250000 /path/to/weights.bin
```

The program first prints every `(family, max_threads, min_blocks, tile, median_ms)` result, then prints the fastest
median for each family. A tile of zero denotes a non-hierarchical family. This keeps device-specific choices outside
the generator and gives users reproducible launch bounds for the inference template arguments and a runtime
hierarchical tile value.

The generator CTest suite runs the MLP autotuner with batch size 100 as a smoke test, covering pool initialization,
weight loading, every emitted configuration, result collection, and clean pool finalization without making routine
test runs pay the default million-sample tuning cost.

## Weight format and loading

`weights.bin` starts with a 32-byte little-endian header: eight-byte `PNNWGT1` magic, format version, scalar code,
payload byte count, and FNV-1a 64-bit payload checksum. The payload is a packed float32 or float64 sequence already in
generated-loop order. `weights.json` records version, scalar type, endianness, tensor names/shapes, byte offsets and
sizes, canonical layouts, model dimensions, and checksum. Compile-time offsets in the header remove any C++ JSON
dependency.

`load_weights()` validates the header, exact file size, scalar metadata, element count, and checksum; converts to the
chosen generated `Scalar`; allocates persistent full-precision and FP16 `ponni::DeviceSpace` views; and deep-copies
them once. Inference does not allocate. The generated object contains only device-safe views and can be copied into a
Kokkos lambda.

## Learned parameter access

Generated models accept only `float` or `double` as their `Scalar`. `get_num_parameters()`, `get_parameters()`, and
`set_parameters()` expose a deterministic flat vector containing only learned parameters. Dense weights and biases,
LayerNormalization scale and bias, and BatchNormalization scale and bias are learned. BatchNormalization running
statistics, shape and reduction axes, clipping bounds, literal ONNX constants, and constants produced by compiler
folding are static and are excluded. `weights.json` records the learned/static classification for every stored tensor.
ONNX has no general `requires_grad` field, so this classification is based on validated operation roles rather than
framework training state.

`get_parameters()` and `set_parameters()` accept rank-one `Kokkos::View`s whose scalar is `float` or `double`; their
extent must equal `get_num_parameters()` in every build mode. `set_parameters()` accepts an execution-space instance,
defaulting to the execution space associated with the source View's memory space, and verifies at compile time that
the execution space can access that memory. It updates the authoritative `Scalar` parameters and always calls
`refresh_half_parameters()` so packed-half inference observes the same update. `parameters_are_finite()` and
`parameters_synchronized()` are explicit, synchronous diagnostics. They are never run implicitly except that
`set_parameters()` checks finiteness when Kokkos is built with `KOKKOS_ENABLE_DEBUG`.

`save_parameters()` writes the complete learned-plus-static state in the validated `weights.bin` format, so the saved
file can be passed back to `load_weights()`. Independently constructed model instances allocate independent parameter
storage and can hold different values. Copying an existing model follows normal `Kokkos::View` shallow-copy semantics:
the copies share that instance's allocations, so changing parameters through one copy changes the others. Applications
must coordinate parameter updates with inference; concurrent mutation and inference on shared storage is unsupported.

## Verification and performance

Compilation compares the unfused IR interpreter to the optimized IR on deterministic random data. Export compares
PyTorch and ONNX Runtime. The operator zoo additionally compares the expanded activation, normalization, probability,
reduction, and math families to ONNX Runtime. C++ tests compare full-precision batched `Kokkos::View`, packed half2,
and embedded `SArray`
inference against PyTorch for small odd and even batch sizes. Failures report output/sample indices, expected and actual
values, and errors. The deeper functionality suite covers depth 10, residual and dense connectivity, branching,
`Relu`/`Tanh`/`Sigmoid`, storage reuse, generalized varying-width dense-chain streaming, and bounded branch
recomputation. A generated-source check ensures the direct
View kernel stages input once, creates no View intermediates, and never materializes dense preactivations.

`generator_benchmark` reports one-time weight loading, warm portable batched targets, and an embedded `infer_one` kernel.
`generator_gpu_scale` provides the GPU-only, large-batch, single-precision comparison used for scheduling decisions.
It sweeps `I -> I -> I -> 3` networks for `I = 4, 8, 16, 32, 64, 128`; batch sizes 10,000, 100,000, and 1,000,000;
and team batch tiles 1 through 32.
Summary rows compare embedded SArray, direct View batch, hierarchical tile 1, the best legal hierarchical tile, the
packed half2 baseline, and optional explicit half2 accumulator policies, including approximate targets' maximum
absolute differences from the FP32 View result.
Kokkos fences bracket every timed device region. For CUDA builds, inspect the compiler output from the
repository's `nvcc_wrapper` flags (for example `--ptxas-options=-v`) for registers and local-memory spills; this is a
diagnostic build choice, not part of generated portable code.

## Extension points

To add an ONNX operator or schema revision, add its reviewed `since_version` to `SUPPORTED_OPERATOR_SCHEMAS`, define
its canonical attributes and precise shape/broadcast restrictions in `importer.py`, implement it in the unfused
interpreter and C++ emitter, then add direct ONNX positive/rejection fixtures and an ONNX Runtime oracle comparison.
Expand the global IR/opset envelope only after all schemas reached by existing supported operators have been reviewed.
Then rerun `examples/probe_onnx_versions.py` to cover real framework decompositions. Do not pass an unvalidated
protobuf node into code generation.

To add a fusion rule, write a deterministic `Graph -> bool` function in `passes.py`, register it in `PASS_PIPELINE`,
require single-consumer/liveness conditions explicitly, add interpreter/emitter support for the new canonical op, and
test both fused code shape and numerical equivalence with the pass disabled.

To add a dense-chain scheduling rule, keep it deterministic in `scheduler.py`, state its legality and cost bounds,
teach the planner about any delayed reads, and test both the emitted source shape and the resulting activation
lifetimes. Streaming must not cross a multiple-consumer value unless it is retained or explicitly recomputed.

To add an execution strategy, add its CLI choice and selection report, then emit an `infer_batch` launcher whose
dependencies and synchronization are valid in Kokkos. Any combined flattened index must keep `ibatch` fastest. Report
local, team-scratch, and external workspace estimates and test both CPU and accelerator builds.

## Known limitations and next steps

This prototype is intentionally an MLP compiler, not a generic ONNX runtime. It currently supports one input/output,
flat, fixed-size per-sample feature storage, constant dense and normalization weights, scalar/exact elementwise
broadcasting, float32/float64
weights, and local or per-team-scratch activation storage. The hierarchical target parallelizes neurons but uses a
serial reduction inside each neuron. Optional team-vector reduction for wide input dimensions is a useful next
scheduling extension. Generalized streaming currently applies only to sample-local
and half2 emission; the hierarchical target continues to materialize the full graph. Convolution, arbitrary rank transposes,
quantization, mixed precision, and multiple inputs/outputs are not implemented.
