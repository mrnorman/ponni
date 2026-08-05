# Understanding and extending the PONNI generator

This tutorial explains the PONNI ONNX-to-Kokkos generator from first principles. It is intended for users who want to
compile a model and for developers who want to understand, debug, or extend the compiler without prior compiler
development experience.

The generator is an **ahead-of-time compiler**. It reads an ONNX model during the build, proves that the model fits a
restricted and well-defined contract, converts it into a smaller internal representation, optimizes that
representation, and writes ordinary C++ plus an external weight file. Python and ONNX are not needed when the
generated C++ runs.

## Contents

1. [The short mental model](#1-the-short-mental-model)
2. [Why PONNI uses a restricted compiler](#2-why-ponni-uses-a-restricted-compiler-instead-of-an-onnx-runtime)
3. [Compiler vocabulary](#3-compiler-vocabulary-used-in-this-project)
4. [Source map](#4-source-map)
5. [Running the generator](#5-running-the-generator)
6. [The compilation phases](#6-phase-one-optional-onnx-preprocessing)
7. [Generated artifacts](#15-generated-artifacts-and-how-to-read-them)
8. [Tests and CMake](#16-how-tests-and-cmake-exercise-the-system)
9. [Debugging](#17-debugging-a-model-or-optimization)
10. [Adding an ONNX operator](#18-adding-support-for-an-onnx-operator)
11. [Adding an optimization](#19-adding-or-changing-an-optimization-pass)
12. [Scheduling and storage innovation](#20-innovating-in-scheduling-or-storage-planning)
13. [Adding a framework exporter](#21-adding-a-framework-exporter)
14. [Correctness rules and reading path](#22-correctness-rules-worth-preserving)

## 1. The short mental model

Think of the generator as a sequence of translations:

```text
PyTorch / Keras / TensorFlow
             |
             | framework exporter
             v
          ONNX model
             |
             | schema validation and import
             v
     PONNI canonical graph
             |
             | deterministic graph rewrites
             v
      optimized PONNI graph
             |
             +--------------------+
             |                    |
             | scheduling         | storage planning
             v                    v
      execution decisions    local arena offsets
             |                    |
             +----------+---------+
                        |
                        | C++ and weight emission
                        v
              GeneratedModel.hpp
              weights.ponni / weights.json
              canonical_ir.json
              optimization_report.json
```

A traditional compiler translates source code into machine code. PONNI translates an ONNX computation graph into a
specialized C++ model class. The model dimensions, operations, parameter offsets, and temporary-storage layout are all
known when the class is generated. Kokkos then compiles that class for the selected CPU or accelerator backend.

## 2. Why PONNI uses a restricted compiler instead of an ONNX runtime

An ONNX runtime must handle many model shapes, operators, devices, and dynamic conditions at runtime. PONNI instead
targets a narrower use case:

- one model input and one model output;
- feature-major tensors with shape `(features, batch)`;
- static per-sample feature dimensions;
- a dynamic batch dimension;
- reviewed `ai.onnx` schemas and semantics;
- inference-only, deterministic graphs;
- generated Kokkos code with no Python or ONNX runtime dependency.

This restriction is useful. It lets the generator reject ambiguity early, remove runtime shape machinery, specialize
loops and storage sizes, expose an inline device function, and keep the generated implementation inspectable.

The authoritative operator and schema envelope is documented in
[ONNX_OPERATOR_SUPPORT.md](../ONNX_OPERATOR_SUPPORT.md). Do not infer support from an operator name alone: ONNX
operator meanings can change between schema versions, and PONNI explicitly reviews those versions.

## 3. Compiler vocabulary used in this project

You only need a few compiler concepts to follow the code:

**Graph**
: A collection of operation nodes connected by tensor edges. A tensor records which node produces it and which nodes
  consume it.

**Intermediate representation (IR)**
: The compiler's private, simplified model of the program. PONNI's IR is not ONNX IR. It is a set of Python data
  classes designed around generated inference.

**Canonicalization**
: Converting different but equivalent source spellings into one common representation. For example, supported ONNX
  `MatMul` and `Gemm` patterns become PONNI `Dense` nodes.

**Optimization pass**
: A deterministic function that rewrites a graph while preserving its result. Passes may remove nodes, fold constants,
  or combine several operations into one compound operation.

**Fusion**
: Combining operations so an intermediate value does not need to be written and read as a separate tensor.

**Liveness**
: The interval from the creation of a tensor until its last use. Two tensors whose live intervals do not overlap may
  share the same storage.

**Scheduling**
: Choosing when and how operations execute. PONNI's scheduler decides when dense activations are materialized,
  streamed directly to a consumer, retained for branches, or recomputed.

**Emission**
: Rendering the final graph and plans as C++ source and binary data.

## 4. Source map

The package is deliberately split by compiler responsibility:

| File | Responsibility |
|---|---|
| [cli.py](cli.py) | Command-line parsing and dispatch |
| [compiler.py](compiler.py) | End-to-end orchestration and report construction |
| [importer.py](importer.py) | ONNX schema resolution, validation, and lowering to PONNI IR |
| [ir.py](ir.py) | Canonical graph, node, tensor, constant, shape, and dtype objects |
| [passes.py](passes.py) | Canonicalization, folding, pruning, and fusion passes |
| [interpreter.py](interpreter.py) | NumPy execution of canonical and optimized graphs |
| [scheduler.py](scheduler.py) | Dense streaming and bounded-recomputation decisions |
| [planner.py](planner.py) | Liveness analysis, in-place aliases, and local arena placement |
| [emitter.py](emitter.py) | Generated Kokkos C++ rendering for all inference APIs |
| [weights.py](weights.py) | PONNI-profile Safetensors serialization and validation |
| [weight_export.py](weight_export.py) | Framework weight adapters and generator ONNX diagnostics |
| [export.py](export.py) | Deterministic PyTorch fixtures and ONNX verification |
| [framework_export.py](framework_export.py) | Keras and TensorFlow fixtures and verification |
| [onnx_reference.py](onnx_reference.py) | CPU ONNX Runtime adapter used by exporters and tests |
| [errors.py](errors.py) | User-facing compiler error type |

The higher-level user reference is [the generator README](../README.md). This tutorial focuses on how and why the
implementation works.

## 5. Running the generator

From the repository root, install the package into a suitable Python environment:

```bash
python -m pip install -e src/generator
```

Validate a model without writing generated code:

```bash
python -m kokkos_nn validate model.onnx
```

Compile a model:

```bash
python -m kokkos_nn compile model.onnx \
  --output-dir generated \
  --model-name MyModel
```

Useful diagnostic commands are:

```bash
# Show the ordered optimization passes.
python -m kokkos_nn list-passes

# Disable one pass to compare the graph before and after that transformation.
python -m kokkos_nn validate model.onnx --disable-pass dense-epilogue-fusion

# Compare native workspace placement with diagnostic alternatives.
python -m kokkos_nn validate model.onnx --analyze-workspace

# Validate a generated PONNI Safetensors file and optional manifest.
python -m kokkos_nn validate-weights generated/weights.ponni \
  --manifest generated/weights.json
```

Both `validate` and `compile` print a JSON report unless `--quiet` is supplied. Start with `validate` while developing:
it exercises import, optimization, scheduling, and planning without writing C++ artifacts.

## 6. Phase one: optional ONNX preprocessing

The default path imports the model exactly as supplied. With `--onnx-preprocess`, [compiler.py](compiler.py) first asks
ONNX Script to perform provider-neutral cleanup such as constant folding, function inlining, shape inference, and dead
code removal.

This preprocessing is deliberately optional:

- producer compatibility remains visible by default;
- PONNI still imports and validates the rewritten model itself;
- the original model remains the reference for reporting and equivalence checks;
- preprocessing is not permission to accept semantics outside PONNI's contract.

When debugging a producer-specific model, compare validation with and without `--onnx-preprocess`. If only the
preprocessed form succeeds, determine whether the source contains a legal spelling the importer should understand or
whether preprocessing is the appropriate user-facing requirement.

## 7. Phase two: ONNX import is the compatibility boundary

[importer.py](importer.py) performs more than parsing. It establishes the semantic facts every later phase may assume.

### 7.1 Version and schema checks

An ONNX model declares an IR version and imports one or more operator-set versions. The importer:

1. checks the ONNX IR version against PONNI's reviewed range;
2. requires a standard `ai.onnx` opset in the reviewed range;
3. preserves domain-specific opset metadata;
4. resolves the exact schema selected for every node;
5. checks that schema against `SUPPORTED_OPERATOR_SCHEMAS`;
6. verifies input/output arity;
7. materializes default attributes from the selected schema;
8. rejects unknown attributes and unsupported domains.

Materializing defaults is important. A later pass should not need to ask whether a missing attribute means “use the
ONNX default” or “the importer forgot it.” The canonical node always carries the reviewed meaning.

### 7.2 Tensor and shape import

Each ONNX value becomes a `TensorValue` with:

- a stable integer ID;
- its source name;
- a shape containing integers and, where legal, a batch `Symbol`;
- a PONNI `DType`;
- producer and consumer links;
- flags for model inputs, outputs, and constants.

The dynamic batch axis is intentionally not included in `TensorValue.sample_shape`. Consequently, `sample_size` is
the exact number of values one generated inference invocation needs for that tensor.

### 7.3 Constants and learned parameters

ONNX initializers and constant nodes become `ConstantTensor` objects. A constant records its values, dtype, canonical
layout, and whether it is a learned parameter. Shape constants may be consumed entirely during import or folding;
learned parameters are later written to `weights.ponni`.

### 7.4 Static shape subgraphs

Framework exporters often emit `Shape`, `Size`, `Gather`, `Squeeze`, `Unsqueeze`, and `Concat` nodes to construct shape
values. If the result depends only on static feature dimensions, PONNI evaluates it at compile time. If it depends on
the runtime batch size in an unsupported way, import fails instead of generating runtime shape code.

### 7.5 Semantic validation

After graph construction, `validate_graph` checks operation-specific invariants. Examples include:

- dense weights must be constant and rank two;
- reductions and normalization must use the complete feature axis;
- broadcasting is limited to reviewed scalar or exact per-sample shapes;
- layout operations must preserve per-sample ordering;
- Boolean comparisons and selections must have compatible dtypes;
- training/stateful behavior and unsupported dynamic shapes are rejected.

This is a crucial design rule: **later phases should not repeatedly defend against invalid ONNX**. They operate on the
smaller, stronger canonical contract established here.

## 8. Phase three: PONNI's canonical IR

[ir.py](ir.py) contains five central concepts:

```text
Graph
  inputs: tensor IDs
  outputs: tensor IDs
  tensors: ID -> TensorValue
  nodes: ordered Node list
  constants: name -> ConstantTensor
  metadata: ONNX provenance and schema counts
```

A `Node` contains an operation name, input/output tensor IDs, canonical attributes, and its original ONNX name. The
IR stays intentionally small: it does not reproduce the complete ONNX object model.

Producer and consumer links are derived data. Any pass that changes nodes or tensor edges must call
`graph.rebuild_links()` or `graph.renumber_nodes()`. Dense, topological node IDs are useful because the planner uses
node positions as liveness coordinates.

### A tiny conceptual example

Suppose ONNX contains:

```text
input -> MatMul(weight) -> Add(bias) -> Relu -> output
```

Immediately after import, the canonical graph is still close to the source:

```text
MatMul -> Add -> Relu
```

After dense canonicalization and fusion it may become:

```text
DenseBiasActivation(activation="Relu")
```

This does not mean the mathematical operations disappeared. They are now represented as one compound node whose
emitter can accumulate, add the bias, apply ReLU, and write the final result without materializing intermediate
vectors.

## 9. Phase four: the optimization pipeline

[passes.py](passes.py) organizes passes into stages. Every stage runs to a fixed point, up to eight iterations. Fixed
point iteration matters because one rewrite can expose another opportunity in the same family. Failure to converge is
a compiler error rather than an invitation to produce an unstable result.

Every pass returns `True` when it changed the graph. The optimization report records whether the pass was disabled,
whether it changed anything, node counts, and iteration counts.

### Stage 1: graph cleanup

`topological-schedule`
: Restores deterministic dependency order.

`constant-fold`
: Executes operations whose inputs are compile-time constants.

`identity-elimination`
: Redirects users around operations that do no work.

`layout-fold`
: Removes static reshape/flatten/squeeze/unsqueeze/transpose scaffolding when per-sample element order is preserved.

`dead-code-elimination`
: Removes nodes and tensors that cannot affect the model output.

### Stage 2: dense canonicalization and structural folding

`dense-canonicalization`
: Converts reviewed `MatMul` and `Gemm` forms into a common `Dense` node with canonical weight orientation.

`dense-gather-pruning`
: When a static gather selects rows of a dense result, prunes unused weight and bias rows instead of computing and
  gathering the full output.

`dense-bias-fusion`
: Moves a compatible bias addition into the dense operation.

`virtual-dense-input-fusion`
: Converts static `Concat` and `Gather` input structure into an `input_map`. The dense loop reads the mapped source
  features directly, avoiding an intermediate concatenated or reordered vector.

### Stage 3: recognize exporter decompositions

Different frameworks may represent one mathematical operation as a small primitive graph. PONNI recognizes reviewed,
exact patterns for:

- SiLU and other supported activations;
- stable Softmax and LogSoftmax;
- decomposed LayerNormalization.

Recognition is deliberately structural and conservative. A graph that merely looks similar is not fused unless the
pattern, constants, axes, and consumer relationships prove equivalent.

### Stage 4: dense and residual fusion

This stage folds common post-dense work into dense nodes:

- activation functions;
- residual addition plus activation;
- dense residual activation;
- ordered epilogues such as normalization and activation sequences.

Ordering matters. Fusion must preserve the source expression order rather than reassociate floating-point arithmetic.

### Stage 5: pointwise and reduction regions

`comparison-where-fusion`
: Combines a comparison mask used only by `Where` into `CompareSelect`, avoiding separate Boolean storage.

`elementwise-chain-fusion`
: Combines a linear elementwise sequence into an ordered program.

`pointwise-region-fusion`
: Combines suitable pointwise DAGs, including reconverging paths, while preserving dependencies.

`mapped-reduction-fusion`
: Streams a fused pointwise map directly into a one-pass reduction where legal.

Compound nodes keep nested step descriptions in attributes. The interpreter and emitter execute those steps in the
recorded order, and `optimized_component_operations` exposes them in reports for auditability.

### Stage 6: final cleanup

The pipeline removes newly dead nodes and restores final topological order. The result is ready for scheduling and
storage planning.

## 10. Phase five: semantic verification with the interpreter

[interpreter.py](interpreter.py) executes both ordinary and fused canonical nodes using NumPy. It is not intended to be
a production runtime. Its purpose is to give the compiler an independent executable meaning for its IR.

During compilation, [compiler.py](compiler.py) creates deterministic random input with seven batch samples, executes
the original imported graph and optimized graph, and compares their outputs. Compilation fails if the difference
exceeds the dtype-appropriate tolerance.

This catches many pass bugs, but it is not a mathematical proof over all possible input values. Each optimization
still needs focused tests for edge cases, shapes, attributes, and graph ownership.

## 11. Phase six: scheduling activations

Fusion changes graph operations. [scheduler.py](scheduler.py) makes a different kind of decision: whether selected
dense activations need storage at all.

Every ordinary activation receives one of four actions:

`materialize`
: Compute the tensor and place it in planned local storage.

`retain`
: Materialize it because multiple consumers or another dependency requires its value later.

`stream`
: Do not store a producer's dense output. Feed its accumulated values directly into one dense consumer.

`recompute`
: Do not retain a shared dense output. Reevaluate its producer separately at selected consumers.

### Dense streaming

For a legal dense pair, streaming can remove a hidden activation vector. PONNI only selects non-overlapping pairs, so
one dense node is not simultaneously claimed by conflicting producer/consumer streams. For small candidate sets the
scheduler scores legal subsets using the actual storage planner; larger paths use a deterministic bounded method.

### Bounded recomputation

Recomputation trades additional multiply-adds for a lower workspace high-water mark. PONNI limits this to reviewed
one-hop dense branches and records the extra multiply-add count. Recursive recomputation is not allowed.

### Aggressiveness levels

| Level | Linear dense streaming | Shared dense branches |
|---:|---|---|
| 1 | Disabled | Materialized |
| 2 | Only a legal pair ending at model output | Materialized |
| 3 | Legal non-overlapping pairs throughout the graph | Materialized |
| 4 | Level 3 around selected branches | Restricted two-consumer terminal recomputation |
| 5 | Level 3 around selected branches | Eligible one-hop recomputation with broader fan-out |

The default is level 3. Levels are deterministic policies, not hardware autotuning settings.

## 12. Phase seven: planning local storage

[planner.py](planner.py) assigns materialized floating tensors and Boolean masks to separate local arenas.

### 12.1 Live intervals

For each eligible tensor, the planner finds:

- `first_use`: the position of its producer;
- `last_use`: the position of its last consumer;
- `size`: the number of per-sample elements.

Streaming and recomputation remove some tensors from storage but extend the liveness of inputs that must be reread at
later consumers.

### 12.2 In-place alias groups

Some operations may overwrite a dead input safely. The planner uses union-find to group compatible input/output
tensors when:

- the input's last consumer is the current node;
- input and output dtypes match;
- the input allocation is large enough;
- the operation has reviewed in-place semantics.

An alias group occupies one arena region sized for its largest member and live for the union of member intervals.

### 12.3 Arena placement

The planner places live intervals at integer offsets. Simultaneously live intervals may not overlap in memory;
non-overlapping lifetimes may reuse offsets.

The native strategy tries several deterministic greedy orders, then performs exact enumeration for up to seven live
groups. `--analyze-workspace` can extend enumeration and optionally use OR-Tools CP-SAT for larger layouts. That exact
path is diagnostic and does not create a runtime dependency.

The important output is the arena **high-water mark**, not the sum of all tensor sizes. Generated inference allocates
only that many local elements.

## 13. Phase eight: weights and parameter layout

[weights.py](weights.py) serializes constant tensors in deterministic lexical-name order. The emitter receives the
corresponding flattened offsets, so physical file order never becomes an accidental code-generation assumption. It
produces:

`weights.ponni`
: A standard Safetensors JSON header and packed little-endian payload. PONNI metadata adds the profile version, exact
  optimized-graph fingerprint, tensor-schema fingerprint, source/target labels, and an FNV-1a checksum over the entire
  payload. Ordinary Safetensors readers can still enumerate and read every tensor.

`weights.json`
: A readable manifest of offsets, shapes, dtypes, learned status, payload size, and checksum.

Generated C++ uses the small dependency-free JSON parser in `src/utils/ponni_json.h`. Before allocating parameter
Views, it validates the profile metadata, exact graph identity, all expected tensor names/dtypes/shapes, overflow-safe
byte lengths, a packed payload with no holes or overlaps, the schema fingerprint, and the payload checksum. It then
stores model-scalar parameters and a synchronized FP16 representation used by `infer_batch_half2`.

The generated parameter API supports loading, saving, inspection, updates, and refreshing the packed-half copy.

## 14. Phase nine: emitting Kokkos C++

[emitter.py](emitter.py) renders one header containing a generated model class. It does not invoke a C++ compiler;
CMake later compiles the header as part of the integration executable or the user's application.

The generated class has `Scalar`, `ExecutionSpace`, and `MemorySpace` template parameters. Execution defaults to
`Kokkos::DefaultExecutionSpace`, and memory defaults to that execution space's native memory. The class checks that
the pair is accessible and stores parameters in ordinary Views in the selected memory space.

The emitter maintains parallel access modes so the same optimized graph can target three APIs:

### `infer_one`

- `KOKKOS_INLINE_FUNCTION`;
- accepts caller-owned `ponni::SArray` input and output;
- intended to be called inside an existing device kernel;
- uses fixed local floating and Boolean arrays determined by the planner.

### `infer_batch`

- accepts nonempty, feature-major `Kokkos::LayoutRight` Views;
- launches a `Kokkos::RangePolicy` over samples;
- stages one sample into local storage;
- performs model-scalar arithmetic.

### `infer_batch_half2`

- accepts nonempty, feature-major `Kokkos::LayoutRight` Views;
- launches over adjacent sample pairs;
- uses `ponni::TwoHalf` and `ponni::TwoMask` values;
- uses persistent FP16 parameters;
- handles an odd final batch sample;
- intentionally has lower-precision semantics than scalar inference.

The emitter has separate read/write helpers for inline scalar, batched scalar, packed-half, and Boolean-mask storage.
Operation-expression helpers are shared where possible so the three paths retain aligned semantics.

Dense streaming decisions change emitted loop nesting: the producer accumulation is consumed directly by the next
dense operation rather than written to a temporary array. Recomputation decisions emit the producer calculation at
each selected consumer.

## 15. Generated artifacts and how to read them

After compilation, inspect these files in this order:

1. `optimization_report.json` — the easiest summary of what happened;
2. `canonical_ir.json` — the complete optimized graph;
3. `weights.json` — parameter names, Safetensors offsets, layouts, and validation metadata;
4. `GeneratedModel.hpp` — the final executable implementation;
5. `weights.ponni` — named tensors inspectable with Safetensors and validated directly or through the CLI.

Useful report fields include:

| Field | Meaning |
|---|---|
| `onnx_opsets` | Imported domain versions |
| `onnx_operator_schema_counts` | Exact schemas selected by the source model |
| `canonical_operations` | Operations immediately after import |
| `optimized_operations` | Top-level operations after all passes |
| `optimized_component_operations` | Operations nested inside fused nodes |
| `passes` | Per-pass change, iteration, and node-count history |
| `storage` | Floating arena plan plus Boolean mask plan |
| `dense_chain_schedule` | Every materialize/retain/stream/recompute decision |
| `fusion_rejections` | Explanations for notable retained dense values |
| `ir_optimization_max_absolute_error` | Deterministic original-versus-optimized check |

## 16. How tests and CMake exercise the system

The unit build in `unit/generator/CMakeLists.txt` has three conceptual phases:

1. export deterministic PyTorch, Keras, and TensorFlow models and reference outputs;
2. compile those ONNX models into headers, weights, IR, and reports;
3. compile and run C++ integration tests against the configured Kokkos backend.

Python tests under `src/generator/tests` cover:

- focused synthetic graph rewrites;
- ONNX version, schema, dtype, shape, and semantic rejection;
- framework export spellings;
- NumPy-versus-ONNX Runtime semantics;
- optional preprocessing and workspace analysis;
- generated operator-document consistency.

C++ integration tests cover all three generated inference APIs, several graph families, workspace policy levels, and
the online parameter API. `check_functionality_generation.py` additionally inspects headers and reports for structural
properties that numerical output alone cannot prove.

Run the complete configured suite with:

```bash
cd unit/build
ctest --output-on-failure
```

When changing Python compiler logic, the focused test is:

```bash
cd unit/build
ctest --output-on-failure -R '^generator_python_test$'
```

Rebuild before C++ tests whenever emitted code or generated artifacts may change.

## 17. Debugging a model or optimization

Use a narrowing workflow:

1. Run `validate` and read the first `CompilerError` literally. Import errors are intended to identify the violated
   contract.
2. Inspect `onnx_opsets` and `onnx_operator_schema_counts` to confirm which semantics the model selected.
3. Compare `canonical_operations` with `optimized_operations`.
4. Find passes with `"changed": true`.
5. Disable the suspected pass and validate again.
6. Inspect `dense_chain_schedule.decisions` if the issue concerns storage or loop structure.
7. Use `--analyze-workspace` if arena fragmentation is the question.
8. Reduce the model to a small synthetic ONNX fixture and add it as a regression test.

Avoid debugging generated C++ first unless import, optimization, interpreter equivalence, and reports are already
correct. Generated code is the last translation of earlier decisions.

## 18. Adding support for an ONNX operator

Adding an operator is a cross-cutting change. Use this checklist:

1. **Study the ONNX schemas.** Determine every schema version selected by PONNI's supported opset envelope. Identify
   defaults, optional inputs, broadcasting, dtype constraints, and changed semantics.
2. **Update the schema registry.** Add only reviewed versions to `SUPPORTED_OPERATOR_SCHEMAS` in `importer.py`.
3. **Import attributes and inputs.** Normalize optional inputs and schema variations into one canonical spelling.
4. **Validate semantics.** Reject shapes, axes, dtypes, modes, or attributes the generator will not implement.
5. **Choose the IR representation.** Reuse an existing canonical operation when meanings match; otherwise introduce a
   clear new operation and attributes.
6. **Implement reference execution.** Add the operation to `interpreter.py`, including edge semantics.
7. **Emit scalar C++.** Add inline/batch handling in `emitter.py`.
8. **Emit packed-half C++.** Add `TwoHalf` or `TwoMask` handling, or explicitly reject an unsupported generated path.
9. **Consider folding and fusion.** Add constant-fold behavior or pointwise-family membership only when safe.
10. **Document restrictions.** Update `examples/generate_operator_support.py` and regenerate the support table.
11. **Add tests.** Include importer rejection tests, ONNX Runtime comparisons, compiler tests, and operator-zoo coverage
    as appropriate.
12. **Build generated C++.** Python success alone does not prove that all emitted paths compile on Kokkos backends.

Regenerate the operator table with:

```bash
PYTHONPATH=src/generator python src/generator/examples/generate_operator_support.py
```

## 19. Adding or changing an optimization pass

A good pass has a narrow legality predicate and a simple transformation.

1. State the exact graph pattern and semantic preconditions.
2. Call `graph.rebuild_links()` before relying on ownership or last-consumer information.
3. Reject ambiguous patterns rather than guessing producer intent.
4. Preserve floating-point evaluation order unless equivalence under reassociation is an explicit contract.
5. Mutate nodes/tensors, then call `renumber_nodes()` when node membership changes.
6. Return whether the graph changed.
7. Insert the pass into the stage where its prerequisites are already canonical and its results can be cleaned up.
8. Ensure repeated execution converges.
9. Add focused positive and negative tests.
10. Compare interpreter results before and after the pass on edge inputs, not only random data.

When a pass creates a compound node, record enough ordered component information for both the interpreter and emitter
to execute it and for reports to explain it.

## 20. Innovating in scheduling or storage planning

Scheduling and placement are related but separate:

- the scheduler decides which values exist and when computations are repeated;
- the planner assigns storage to values that remain materialized.

For a scheduling experiment:

1. define legal candidates independently of the cost model;
2. include all extended input lifetimes caused by streaming or recomputation;
3. prevent overlapping transformations from claiming the same node;
4. score the complete planned high-water mark, not only the size of one eliminated tensor;
5. use deterministic tie-breakers;
6. bound exhaustive searches and provide a stable fallback;
7. report added arithmetic and eliminated storage explicitly;
8. update the emitter and structural C++ tests together.

For a placement experiment, preserve the no-overlap rule for simultaneous live intervals and compare against the
existing heuristic/native/exact diagnostic results. A smaller sum of individual offsets is irrelevant if the arena
high-water mark does not improve.

## 21. Adding a framework exporter

Framework model exporters are test/build conveniences around the ONNX contract. A new model exporter should:

1. force CPU-only export behavior before importing the framework;
2. create deterministic parameters and inputs;
3. normalize boundaries to `(features, batch)`;
4. attach PONNI orientation, batch-symbol, and exporter metadata;
5. validate the ONNX model;
6. compare framework output with CPU ONNX Runtime across several batch sizes;
7. write a reference file consumed by generated C++ tests;
8. add explicit CMake outputs and dependencies;
9. add Python and generated-C++ integration coverage.

Exporter-specific workarounds should stay in exporter code. The importer should accept models based on ONNX semantics,
not on producer identity.

Separately, [weight_export.py](weight_export.py) contains small adapters for Keras, TensorFlow, PyTorch, JAX/Flax,
scikit-learn MLPs, and PaddlePaddle. They use duck typing and lazy framework access, so importing the generator does not
install or import every training framework. A generator-oriented adapter call must receive `onnx_path`; it invokes the
same `validate_model()` compatibility boundary used by the CLI. A rejection includes the original PONNI diagnostic,
IR/opset versions, operator counts, boundary shapes/types, and named node inputs/outputs. Keep this compatibility test
in the generator suite: template mode consumes a narrow explicit layer tuple and cannot make a truthful claim about
support for an ONNX graph.

## 22. Correctness rules worth preserving

- ONNX schema version is part of an operation's identity.
- Import establishes strong invariants; later phases should consume them.
- Feature-major orientation and the symbolic batch axis must remain explicit.
- Public PONNI Views use `Kokkos::LayoutRight`; callers copy unsupported layouts before inference.
- Batch inference requires at least one sample.
- A graph rewrite must preserve dependencies and evaluation order.
- Producer/consumer links must be rebuilt after rewrites.
- Host code must not directly access device Views.
- Device lambdas must capture only device-safe values and Views.
- Floating and Boolean local arenas remain separate.
- Packed-half inference is a distinct numerical path and needs direct tests.
- Scheduling cost models must include lifetime extensions and recomputation work.
- Generated artifacts and reports are part of the observable compiler interface.
- Determinism is preferred over architecture-specific tuning in this generator.

## 23. Suggested reading path for a new developer

Read the implementation in this order:

1. [ir.py](ir.py) — learn the data model;
2. [compiler.py](compiler.py) — see the complete sequence;
3. [importer.py](importer.py) — understand the accepted contract;
4. the `PASS_STAGES` list in [passes.py](passes.py) — learn transformation order;
5. one small pass such as identity elimination, then one fusion pass;
6. [interpreter.py](interpreter.py) — see canonical operation meanings;
7. [scheduler.py](scheduler.py) and [planner.py](planner.py) — understand workspace decisions;
8. the access helpers and one operation family in [emitter.py](emitter.py);
9. [weights.py](weights.py) — understand generated parameter storage;
10. focused tests, then CMake and C++ integration tests.

Do not begin by reading `emitter.py` from top to bottom. It is necessarily large because it renders three execution
paths and many operations. First understand the IR node and plan that a particular emitter branch consumes.

## 24. A practical first extension exercise

A safe way to learn the system is to add a small optimization report field rather than a new operator:

1. choose a fact already present in the optimized graph;
2. compute it in `_report` in `compiler.py`;
3. add a focused assertion in `test_compiler.py`;
4. run the generator Python test;
5. rebuild generated models;
6. inspect `optimization_report.json`;
7. run the complete CTest suite.

Next, modify or add a simple canonicalization with a synthetic two- or three-node model. Only after those exercises
should you add an ONNX operator or a new emitted execution strategy.

The central principle is simple: each phase should make the next phase easier to reason about. Import turns broad ONNX
semantics into a strict contract. Passes turn many graph spellings into a few canonical operations. Scheduling and
planning turn graph lifetimes into explicit execution and storage decisions. Emission then becomes a mechanical,
auditable translation of those decisions into Kokkos C++.
