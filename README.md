# PONNI: POrtable Neural Network Inferencing

PONNI targets a specific HPC niche: **small neural networks evaluated over very large batches of independent
samples**. Typical examples include applying the same compact network to every grid column, cell, particle, ensemble
member, or other local state inside a larger simulation. PONNI uses Kokkos so the same inference code can run on the
CPU and accelerator backends supported by the surrounding application.

PONNI is not intended to be a general runtime for large language models, dynamic sequence models, or arbitrary ONNX
graphs. Its strength is transparent, portable C++ inference for networks small enough to embed directly in an HPC
workflow, while the batch or outer simulation provides abundant parallelism.

Author: Matt Norman, Oak Ridge National Laboratory, <https://mrnorman.github.io>

## Two ways to use PONNI

PONNI supports two complementary workflows:

1. **Template C++ mode:** construct a model directly from PONNI layer templates such as `Matvec`, `Bias`, and `Relu`.
   This is the simplest option when the architecture is naturally expressed in C++ or the application already owns
   the weights.
2. **Ahead-of-time generator mode:** export a model to ONNX, then generate a specialized Kokkos model struct and a
   validated PONNI Safetensors file. This supports a broader graph of dense, residual, branched, normalization, reduction,
   activation, and elementwise operations.

Both modes use nonempty, feature-major arrays: the first dimension is the feature index and the second is the batch
index. Every View passed to PONNI must declare `Kokkos::LayoutRight`. `LayoutLeft` and `LayoutStride` are rejected at
compile time; applications must copy data into a `LayoutRight` View before inference.

## Mode 1: construct a model with C++ templates

Include `ponni.h`, allocate ordinary Kokkos Views, and list the layers in execution order. By default, PONNI uses
`Kokkos::DefaultExecutionSpace` and that execution space's native memory. The example below creates a `2 -> 3 -> 1`
network and evaluates a large batch with one Kokkos iteration per sample.

```cpp
#include "ponni.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
  Kokkos::initialize(argc, argv);
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceMatrix = Kokkos::View<float**, Kokkos::LayoutRight, MemorySpace>;
    using DeviceVector = Kokkos::View<float*, Kokkos::LayoutRight, MemorySpace>;

    int constexpr batch_size = 1000000;

    DeviceMatrix weights_1("weights_1", 2, 3);
    DeviceVector bias_1("bias_1", 3);
    DeviceMatrix weights_2("weights_2", 3, 1);
    DeviceVector bias_2("bias_2", 1);

    // Replace these constants with application or file-loaded parameters.
    Kokkos::deep_copy(weights_1, 0.25f);
    Kokkos::deep_copy(bias_1, 0.10f);
    Kokkos::deep_copy(weights_2, 0.50f);
    Kokkos::deep_copy(bias_2, -0.20f);

    auto model = ponni::create_inference_model(
        ponni::Matvec<float>(weights_1),
        ponni::Bias<float>(bias_1),
        ponni::Relu<float>(3),
        ponni::Matvec<float>(weights_2),
        ponni::Bias<float>(bias_2));
    model.validate();

    DeviceMatrix inputs("inputs", 2, batch_size);
    Kokkos::deep_copy(inputs, 1.0f);

    // Scratch storage grows automatically and is retained for later batches.
    DeviceMatrix outputs = model.forward_batch_parallel(inputs);
    auto const outputs_host = ponni::create_host_copy(outputs);
    std::cout << "first prediction: " << outputs_host(0,0) << '\n';
  }
  Kokkos::finalize();
  return 0;
}
```

Template mode also provides layers for residual and concatenation graphs, normalization, and common activation
functions. Layer parameters can be serialized, restored, and updated through the model API. `save_weights()` writes
one flattened `parameters` tensor and `load_weights()` requires the exact tuple-derived layer fingerprint, dtype, and
parameter count before modifying the model:

```cpp
std::string error;
if (!model.save_weights("template_model.ponni", &error)) throw std::runtime_error(error);
if (!model.load_weights("template_model.ponni", &error)) throw std::runtime_error(error);
```

This template file contract intentionally validates the explicit C++ layer tuple; ONNX compatibility is validated
only by the generator workflow. Model-owned parameters, saved states, and temporary Views use the model's selected
Kokkos memory space.

To select a different execution instance and accessible memory space, pass them before the layers. The factory
rebinds every layer and its parameters, so the policy is specified once:

```cpp
Kokkos::DefaultHostExecutionSpace host_execution;
Kokkos::HostSpace host_memory;

auto host_model = ponni::create_inference_model(
    host_execution,
    host_memory,
    ponni::Matvec<float>(weights_1),
    ponni::Bias<float>(bias_1),
    ponni::Relu<float>(3));
```

Every layer has a trailing `MemorySpace` template parameter that defaults to the default execution space's native
memory. Applications normally let `create_inference_model` rebind it. The model stores the supplied execution-space
instance, preserving custom streams, and launches its `RangePolicy` on that instance. Internal batch storage grows
only when a larger batch arrives; `reallocate_internal_state(batch_size)` performs an exact resize when an application
wants to shrink retained capacity or prepare the View-based intra-kernel path.

A custom layer that participates in factory rebinding must provide both
`rebind_memory_space<NewMemorySpace>` and `copy_to_memory_space(NewMemorySpace const&)`. The copy operation returns the
rebound layer, preserves scalar configuration, and copies every layer-owned View with
`ponni::create_memory_space_copy`. This is a direct layer copy; weight persistence is handled separately by the
validated `.ponni` APIs.

Inference requires `batch_size > 0`. Passing a zero-column input is an error. A zero value remains valid for
`reallocate_internal_state(0)`, which explicitly releases retained internal storage without launching inference.

## Mode 2: generate a specialized Kokkos model from ONNX

Install the generator from the repository root, validate the ONNX contract, and generate the C++ model:

```bash
python -m pip install -e src/generator

python -m kokkos_nn validate model.onnx

python -m kokkos_nn compile model.onnx \
  --output-dir generated \
  --model-name MyModel
```

The output directory contains:

- `MyModel.hpp`: the specialized Kokkos model class;
- `weights.ponni`: named Safetensors plus PONNI graph/schema fingerprints and a payload checksum;
- `weights.json`: a readable parameter manifest;
- `canonical_ir.json`: the optimized compiler graph;
- `optimization_report.json`: passes, fusions, storage, scheduling decisions, and verification results.

Generated C++ depends only on PONNI and Kokkos. Python, ONNX, and the source framework are build-time tools.
The C++ loader uses PONNI's dependency-free JSON reader, checks every expected tensor name/dtype/shape and byte range,
requires the exact generated-graph fingerprint, and verifies the payload checksum before copying parameters to the
selected Kokkos memory space. Standard Safetensors tools can inspect the same `.ponni` file.

### Use the generated model inside an existing kernel

`infer_one` is a `KOKKOS_INLINE_FUNCTION` that accepts fixed-size `ponni::SArray` values. Load weights on the host
before launching the kernel, then capture the model's parameter Views by value. The fragment below belongs inside an
initialized Kokkos scope, like the scope in the template-mode example.

```cpp
#include "MyModel.hpp"

#include <stdexcept>
#include <string>

using Model = ponni::generated::MyModel<float>;
int constexpr batch_size = 1000000;

Model model;
std::string error;
if (!model.load_weights("generated/weights.ponni", &error)) {
  throw std::runtime_error(error);
}

Model::InputView inputs("inputs", Model::num_inputs, batch_size);
Model::OutputView outputs("outputs", Model::num_outputs, batch_size);

// Initialize inputs through a host mirror; do not index GPU-resident memory on the host.
auto inputs_host = Kokkos::create_mirror_view(inputs);
for (int ibatch = 0; ibatch < batch_size; ibatch++) {
  for (int i = 0; i < Model::num_inputs; i++) {
    inputs_host(i,ibatch) = static_cast<float>(i + ibatch % 4);
  }
}
Kokkos::deep_copy(inputs, inputs_host);

auto const device_model = model;
Kokkos::parallel_for("application_with_embedded_nn", batch_size, KOKKOS_LAMBDA(int ibatch) {
  ponni::SArray<float, Model::num_inputs> sample_inputs;
  ponni::SArray<float, Model::num_outputs> sample_outputs;

  for (int i = 0; i < Model::num_inputs; i++) {
    sample_inputs(i) = inputs(i,ibatch);
  }
  device_model.infer_one(sample_inputs, sample_outputs);
  for (int i = 0; i < Model::num_outputs; i++) {
    outputs(i,ibatch) = sample_outputs(i);
  }
});
```

This is the intended path when inference is one step inside a larger application kernel. The application controls the
outer launch and may combine model inputs or outputs with its other per-sample calculations.

### Let the generated model launch a standalone batch kernel

For ordinary batched inference, allocate feature-major Views and call `infer_batch`. This is the vanilla generated
batch API; it launches one Kokkos iteration per sample.

```cpp
#include "MyModel.hpp"

#include <stdexcept>
#include <string>

using Model = ponni::generated::MyModel<float>;
int constexpr batch_size = 1000000;

Model model;
std::string error;
if (!model.load_weights("generated/weights.ponni", &error)) {
  throw std::runtime_error(error);
}

Model::InputView inputs("inputs", Model::num_inputs, batch_size);
Model::OutputView outputs("outputs", Model::num_outputs, batch_size);

auto inputs_host = Kokkos::create_mirror_view(inputs);
for (int ibatch = 0; ibatch < batch_size; ibatch++) {
  for (int i = 0; i < Model::num_inputs; i++) {
    inputs_host(i,ibatch) = static_cast<float>(i + ibatch % 4);
  }
}
Kokkos::deep_copy(inputs, inputs_host);

model.infer_batch(inputs, outputs);

auto const outputs_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs);
// Read outputs_host(feature, sample) or pass it to the application's next step.
```

Call `load_weights` only after `Kokkos::initialize` and before inference. Keep the model alive while kernels using its
parameter Views are in flight.

## Include PONNI in a CMake project

PONNI defines a `ponni` target that publicly links `Kokkos::kokkos` and publishes the PONNI include directories. Add
Kokkos first, then PONNI, and link the application against `ponni`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(MyApplication LANGUAGES CXX)

# These may also be provided through find_package() in an installed workflow.
add_subdirectory(/path/to/kokkos kokkos)
add_subdirectory(/path/to/ponni ponni)

add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE ponni)

# Add this only when main.cpp includes a generated header such as MyModel.hpp.
target_include_directories(my_application PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/generated)
```

For template mode, `#include "ponni.h"` is sufficient. For generator mode, also include the generated model header and
make its output directory visible to the target. The generated header includes `ponni.h` itself, but the application
still links `ponni` to inherit PONNI's headers and the selected Kokkos backend.

## Build and run the unit tests

The repository carries Kokkos as a test submodule and provides machine profiles under `unit/build/machines`. From a
fresh clone:

```bash
git clone git@github.com:mrnorman/ponni.git
cd ponni
git submodule update --init --checkout -- unit/externals/kokkos

cd unit/build

# Select the compiler, Kokkos backend, architecture, and debug/coverage settings.
source machines/thatchroof/thatchroof_cpu_coverage.env

./cmakescript.sh
cmake --build . -j8
ctest --output-on-failure
```

Choose a different profile under `unit/build/machines`, or create one for the local platform. The sourced profile sets
`KOKKOS_HOME`, compilers, backend and architecture options, and PONNI compile flags.

The build creates a repository-owned `uv` installation and CPU-only Python environment under `unit/build`; it does not
install framework packages into the user's home environment. Python produces framework reference data and ONNX models,
while the configured Kokkos backend compiles and runs the C++ tests. The performance benchmark is built but is not
registered as a CTest test.

See [unit/README.md](unit/README.md) for GPU-debug profiles, generator-test details, Python dependency behavior, and
the gcov/gcovr coverage workflow.

## More documentation

- [Generator guide](src/generator/README.md): supported graph shape, CLI options, workspace-reduction levels,
  generated APIs, weight management, and verification behavior.
- [Generator tutorial](src/generator/kokkos_nn/TUTORIAL.md): a from-scratch explanation of ONNX import, the canonical
  IR, optimization passes, scheduling, storage planning, C++ emission, debugging, testing, and extension workflows.
- [ONNX operator support](src/generator/ONNX_OPERATOR_SUPPORT.md): authoritative reviewed opsets, operator schemas,
  restrictions, and unsupported standard operators.
- [Unit testing guide](unit/README.md): machine profiles, Python-generated test data, CTest coverage, generated
  artifacts, and coverage reporting.

The template layer API includes dense matrix-vector operations, bias, residual-state save/add operations, feature
concatenation and projection, normalization, and these activation families:

- `Relu`, `LeakyRelu`, `Elu`, and `Selu`;
- `Gelu`, `Silu`, `Sigmoid`, and `Tanh`;
- `Softmax`, `LogSoftmax`, and `Softplus`;
- `HardSigmoid`, `HardSwish`, and `Mish`.
