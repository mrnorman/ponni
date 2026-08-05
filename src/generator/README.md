# PONNI Kokkos neural-network generator

PONNI compiles a constrained, fixed-feature ONNX inference graph into deterministic Kokkos C++ and a validated
PONNI-profile Safetensors file. Python and ONNX are build-time dependencies only. Generated inference depends on
PONNI and Kokkos.

The complete operator matrix is generated in [ONNX_OPERATOR_SUPPORT.md](ONNX_OPERATOR_SUPPORT.md). It records the
reviewed ONNX schema range, supported restrictions, and unsupported operators.

For a beginner-oriented walkthrough of the compiler pipeline, internal representation, optimization passes, storage
planning, generated Kokkos APIs, testing strategy, and extension workflow, see
[kokkos_nn/TUTORIAL.md](kokkos_nn/TUTORIAL.md).

## Pipeline

The compiler imports ONNX into PONNI's canonical graph, validates it, applies deterministic folding and fusion passes,
minimizes activation lifetimes, and assigns remaining floating-point and Boolean intermediates to reusable local
slots. It writes a generated header, `weights.ponni`, `weights.json`, `canonical_ir.json`, and
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

### End-to-end trained-model example

The following small PyTorch example defines and trains a network using ordinary sample-major framework tensors. Its
`forward()` method transposes the boundary tensors so the exported ONNX model follows PONNI's `(features, batch)`
contract. The standalone weight export is a portable, named-tensor checkpoint and also validates that the exported
ONNX graph is supported by PONNI.

```python
import onnx
import torch

from kokkos_nn import export_pytorch_weights


class SmallBatchNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, feature_batch):
        return self.network(feature_batch.transpose(0, 1)).transpose(0, 1)


model = SmallBatchNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-2)
training_inputs = torch.randn(4, 64)
training_targets = torch.stack((
    training_inputs[0] + 0.5 * training_inputs[1],
    torch.tanh(training_inputs[2] - training_inputs[3]),
))
for _ in range(200):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(training_inputs), training_targets)
    loss.backward()
    optimizer.step()

model.eval()
example = torch.zeros(4, 3)
batch_dimension = torch.export.Dim("batch", min=1, max=4096)
torch.onnx.export(
    model,
    args=(example,),
    f="small_batch.onnx",
    dynamo=True,
    optimize=True,
    verify=True,
    input_names=["features"],
    output_names=["predictions"],
    dynamic_shapes=({1: batch_dimension},),
)

# Record the feature-major boundary contract expected by the generator.
onnx_model = onnx.load("small_batch.onnx")
for key, value in {
    "ponni.orientation": "features_batch",
    "ponni.batch_symbol": "batch",
}.items():
    entry = onnx_model.metadata_props.add()
    entry.key = key
    entry.value = value
onnx.save(onnx_model, "small_batch.onnx")

export_pytorch_weights(
    model,
    "trained_weights.ponni",
    onnx_path="small_batch.onnx",
)
```

Compile the validated ONNX graph into a Kokkos struct and its exact, fingerprinted weight file:

```bash
python -m kokkos_nn compile small_batch.onnx \
  --output-dir generated --model-name SmallBatchModel
```

The generated struct must load `generated/weights.ponni`. That file contains the same trained values as the ONNX
initializers after PONNI has canonicalized their names, shapes, and layouts. The generic `trained_weights.ponni`
checkpoint is useful for inspection and interchange, but it deliberately does not impersonate the generated graph's
fingerprint.

Use `infer_one` inside an existing device kernel, or use `infer_batch` when PONNI should own the batch launch:

```cpp
#include "generated/SmallBatchModel.hpp"

#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  using Model = ponni::generated::SmallBatchModel<float>;

  Model model;
  std::string error;
  if (!model.load_weights("generated/weights.ponni",&error)) {
    throw std::runtime_error("Unable to load PONNI weights: " + error);
  }

  int constexpr batch_size = 32;
  Model::InputView inputs("inputs",Model::num_inputs,batch_size);
  Model::OutputView batch_outputs("batch_outputs",Model::num_outputs,batch_size);
  Model::OutputView inline_outputs("inline_outputs",Model::num_outputs,batch_size);

  auto inputs_host = Kokkos::create_mirror_view(inputs);
  for (int i = 0; i < Model::num_inputs; i++) {
    for (int ibatch = 0; ibatch < batch_size; ibatch++) {
      inputs_host(i,ibatch) = static_cast<float>(i + ibatch) / 32.f;
    }
  }
  Kokkos::deep_copy(inputs,inputs_host);

  // Standalone launch: PONNI parallelizes over the batch dimension.
  model.infer_batch(inputs,batch_outputs);

  // Intra-kernel use: each caller iteration owns one fixed-size sample.
  auto const device_model = model;
  Kokkos::parallel_for("application_kernel",batch_size,KOKKOS_LAMBDA(int ibatch) {
    ponni::SArray<float,Model::num_inputs> sample_inputs;
    ponni::SArray<float,Model::num_outputs> sample_outputs;
    for (int i = 0; i < Model::num_inputs; i++) sample_inputs(i) = inputs(i,ibatch);
    device_model.infer_one(sample_inputs,sample_outputs);
    for (int i = 0; i < Model::num_outputs; i++) inline_outputs(i,ibatch) = sample_outputs(i);
  });
  Kokkos::fence();
  return 0;
}
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

Batch inputs, outputs, and parameter-transfer Views must explicitly use `Kokkos::LayoutRight`, and inference requires
at least one batch sample. Generated code rejects `LayoutLeft` and `LayoutStride` at compile time and aborts a
zero-sized batch. Copy external data into a nonempty `LayoutRight` View before calling a generated API.

All three paths support the complete operator set described in `ONNX_OPERATOR_SUPPORT.md`. Boolean intermediates use
compact byte or `TwoMask` local storage. No generated inference API requests Kokkos team scratch.

Generated classes default to `Kokkos::DefaultExecutionSpace` and its native memory space. Both are template
parameters, so an application may select another accessible pair once for its generated View aliases and parameters:

```cpp
using Model = ponni::generated::MyModel<
    float,
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::HostSpace>;
```

Generated models store their parameters in ordinary `Kokkos::LayoutRight` Views in the selected memory space.

## Weights and learned parameters

`weights.ponni` is an ordinary Safetensors container with a `.ponni` extension. Standard Safetensors tools can inspect
its named tensors. PONNI adds string metadata for the profile version, exact generated-graph fingerprint, tensor-schema
fingerprint, source framework, target, and an FNV-1a checksum of the complete payload. `weights.json` mirrors tensor
offsets, shapes, canonical layouts, learned status, and validation metadata for humans and build tooling.

`load_weights()` checks the JSON structure, complete and non-overlapping payload layout, dtype and shape of every
expected tensor, graph and schema fingerprints, and payload checksum before creating persistent scalar and FP16 Views
in the model's memory space. `get_parameters()`, `set_parameters()`, `save_parameters()`, and
`refresh_half_parameters()` support online parameter updates while keeping both representations synchronized.

The `kokkos_nn.weight_export` module also provides lazy, dependency-neutral adapters for Keras, TensorFlow, PyTorch,
JAX/Flax, scikit-learn MLPs, and PaddlePaddle. In generator-oriented use, pass the exported ONNX path; the adapter runs
PONNI's generator validator before writing and reports the failing PONNI rule, opsets, operator inventory, model
boundaries, and named nodes when the graph is unsupported:

```python
from kokkos_nn import export_pytorch_weights

report = export_pytorch_weights(
    trained_model,
    "model_weights.ponni",
    onnx_path="model.onnx",
)
```

These generic named-tensor exports do not replace `kokkos_nn compile`: the compiler remains responsible for
canonicalizing ONNX tensor layouts and writing the exact `weights.ponni` accepted by its generated struct. Supplying a
templated-model fingerprint is an explicit weight-only path and therefore does not claim ONNX compatibility.

## Verification

Compilation runs the original and optimized canonical graphs on deterministic inputs and rejects an optimization if
their results disagree. Unit tests also compare exported models against CPU ONNX Runtime, compile generated headers,
and exercise all three inference APIs across PyTorch, Keras, TensorFlow, and operator-zoo examples.

The unit-test Python environment is unified and CPU-only because ONNX emission and reference semantics are independent
of the Kokkos backend. CUDA or HIP is needed only when generated C++ tests are configured for those backends.
