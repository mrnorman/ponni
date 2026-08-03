# Unit testing

Workflow to run the unit tests:

```bash
git clone git@github.com:mrnorman/ponni.git
cd ponni
# Run command below to use the kokkos submodule of this repo
git submodule update --init --checkout -- unit/build/externals/kokkos
cd unit/build
# To change the kokkos used for testing, change KOKKOS_HOME in the machine file
source machines/[machine_name]/[machine_file]
./cmakescript.sh
make -j8
ctest -V
```

For a clean GPU debug build on `thatchroof`, run:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_debug.env
./cmakescript.sh
make -j
ctest -V
```

All tests are registered with CTest except `performance_benchmark`, which is built but not run by `ctest`.

## Activation coverage

`core_unit_test` exercises every activation layer:

- `Relu`, `LeakyRelu`, `Elu`, `Selu`, `Gelu` (exact and approximate), `Silu`, `Sigmoid`, and `Tanh`
- `Softmax`, `LogSoftmax`, `Softplus`, `HardSigmoid`, `HardSwish`, and `Mish`

For each activation, the test covers validation and metadata, the zero-trainable-parameter API, `to_array()` and
`from_array()` round trips, fixed-size `SArray` evaluation, and `DeviceSpace` `Kokkos::View` evaluation. This makes the
GPU debug profile useful for detecting accidental host access to device memory.

## Python-generated test data (Keras/PyTorch tests)

The following tests generate their HDF5 data at build time via Python scripts:

- `keras_sequential_test`
- `keras_resnet_test`
- `pytorch_resnet_test`

Build-time behavior:

- Python 3.x is discovered via `find_package(Python3 ...)`.
- `uv` is resolved during the **make phase**.
- If `uv` is missing, it is installed in the build tree at `unit/build/uv_env`.
- A local venv is created in `unit/build/python_env`.
- Python dependencies are installed into that venv: `torch`, `keras`, `tensorflow`, `tf2onnx`, `numpy`, `h5py`,
  `onnx`, `onnxruntime`, and `onnxscript`.
- Generator scripts produce HDF5 files in each test's build directory.

This workflow keeps uv and Python dependencies inside `ponni/unit/build` and does not install files under `~/.local`.

No Python package installation is performed during CMake configure.

## Ahead-of-time generator tests

The build exports deterministic PyTorch shallow MLP/residual examples plus depth-10, ResNet, DenseNet, branched,
and operator-zoo functionality models. It also exports a Keras MLP and normalization pipeline through
`Model.export(format="onnx")` and a pure TensorFlow residual module through `tf2onnx`. All are compiled into Kokkos
headers, and the build creates
`generator_integration` and `generator_benchmark`. Run the focused tests and benchmark with:

```bash
cd unit/build
ctest -V -R generator
./generator/generator_benchmark generator/generated/mlp_generated/weights.bin
```

`generator_python_test` covers PyTorch, Keras, and TensorFlow ONNX interchange; Keras no-bias dense layers,
branching, concatenation, residuals, direct activation attributes, decomposed normalization, and the currently
unsupported Boolean-select ELU spelling; TensorFlow transposed constant weights,
`bias_add`, reshape, shared branches, and unsupported-op diagnostics; importer diagnostics;
canonicalization; fusion; generalized dense-chain scheduling;
bounded branch recomputation, liveness/storage reuse, weight
validation, unfused-versus-optimized IR numerics, and ONNX Runtime comparisons for expanded activations,
normalization, softmax/reduction, and math operators. `generator_integration_test` runs generated batched `DeviceSpace`
batch-only, hierarchical tile 1/default, packed half2, embedded `SArray`, and eligible CUDA Tensor Core inference.
Together these cover the five inference families: inline SArray, View batch, hierarchical team-neuron, raw-CUDA
TF32 WMMA, and packed two-sample FP16. Half2 tests include the baseline single dependency chain, the generated
per-dense heuristic, and an explicit four-partial policy; the compiler unit tests cover every accepted explicit count
(`0`, `2`, `4`, `8`, `16`, and `32`) and reject other values.
The diverse models use batches 1, 2, 3, 7, and 11; the original examples retain 1, 2, 7, 32, and 67. A separate
structure test rejects generated input rereads, View intermediates, truncated live workspaces, and preactivation arrays,
and requires the operator-zoo operations to survive into generated code.

When CUDA is enabled, `generator_gpu_scale_test` additionally checks `I -> I -> I -> 3` networks for
`I = 4, 8, 16, 32, 64, 128`; team batch tiles 1, 2, 4, 8, 16, and 32; and single-precision batches 10,000,
100,000, and 1,000,000 using a 1 GiB device pool. The hierarchical loop orders batch fastest within each neuron and
uses batch-fastest scratch. Eligible models additionally test a generated raw-CUDA WMMA TF32 kernel without Kokkos
hierarchical-policy overhead. Use the debug machine file
for correctness, but use
`machines/thatchroof/thatchroof_gpu_fast.env` for walltimes used to evaluate scheduling changes. The summary reports
SArray, direct View batch, hierarchical tile 1, the best hierarchical tile, Tensor Core TF32, and Kokkos-launched
packed half2 with baseline, generated-heuristic, and explicit four-accumulator policies. The Tensor Core sweep
measures all legal choices among 1, 2, 4, and 8 warps per CUDA block; current
Ampere defaults select 4, 2, 2,
4, 2, and 1 warps for I=4, 8, 16, 32, 64, and 128, respectively. The hierarchical default is tile 32. Sample-local
emission deterministically selects non-overlapping dense pairs throughout deeper chains, streaming each
selected activation into scalar output accumulators and retaining or cheaply recomputing branches as reported.

## gcov coverage workflow

To build unit tests with gcov instrumentation and generate coverage reports:

```bash
cd unit/build
# Use a machine file, then enable coverage before configuring.
source machines/[machine_name]/[machine_file]
export PONNI_COVERAGE=ON
./cmakescript.sh
make -j8
# Run all tests with verbose output.
ctest -V
# Generate coverage report target (requires gcov; gcovr is optional).
make coverage
```

Coverage output:
- `unit/build/coverage/gcov.txt` (always generated when `gcov` is available)
- `unit/build/coverage/coverage.txt` (generated when `gcovr` is available)
- `unit/build/coverage/index.html` (generated when `gcovr` is available)

`ctest -V` also includes `src_gcov_summary_test`, which prints a recursive per-file summary for `ponni/src` and writes:

- `unit/build/coverage/src_gcov_summary.txt`
- `unit/build/coverage/gcov_intermediate_all.txt`
