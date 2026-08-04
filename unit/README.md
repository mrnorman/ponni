# Unit testing

Workflow to run the unit tests:

```bash
git clone git@github.com:mrnorman/ponni.git
cd ponni
# Run command below to use the kokkos submodule of this repo
git submodule update --init --checkout -- unit/externals/kokkos
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
- A unified CPU-only venv is created in `unit/build/python_cpu_env`, independently of the configured Kokkos backend.
- Python dependencies are installed into that venv: `torch`, `keras`, `tensorflow`, `tf2onnx`, `numpy`, `h5py`,
  `onnx`, `onnxruntime`, and `onnxscript`.
- Generator scripts produce HDF5 files in each test's build directory.

The environment uses Python 3.11 or newer, CPU PyTorch and TensorFlow, and `onnxruntime>=1.25,<2`. Frameworks only
export and validate backend-neutral ONNX; CUDA/HIP correctness is exercised separately by the generated Kokkos C++.
This workflow keeps uv and Python dependencies inside `ponni/unit/build` and does not install files under `~/.local`.

No Python package installation is performed during CMake configure.

## Ahead-of-time generator tests

The generator tests use one CPU-only Python environment at unit/build/python_cpu_env for PyTorch, Keras,
TensorFlow, ONNX, and ONNX Runtime. Configure and build normally; the ponni_python_env target creates or refreshes it.

The integration tests export representative framework and operator-zoo models, compile them to Kokkos C++, and
compare infer_one, infer_batch, and infer_batch_half2 with CPU reference data. Structure checks verify that all three
APIs are present and that obsolete team policies, team scratch, and batch-team entry points are absent.

Generated artifacts are under unit/build/generator/generated. Each model directory contains its header, weights,
canonical_ir.json, and optimization_report.json. Generator performance experiments and architecture-specific
autotuning are intentionally not part of the unit suite.

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
