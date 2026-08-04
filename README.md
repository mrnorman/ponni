# PONNI: POrtable Neural Network Inferencing
### Efficient in-loop neural network inferencing made easy in C++

Author: Matt Norman, Oak Ridge National Laboratory, https://mrnorman.github.io

PONNI provides a convenient way to build an efficient, portable Neural Network inference model in C++ with minimal syntax and full disclosure of exactly how the model is running on an accelerator device. It uses the Kokkos portable C++ library.

## Using PONNI in your CMake project
```CMake
cmake_minimum_required(VERSION 3.22)
project(MyProject)
add_subdirectory(/path/to/kokkos kokkos) # or use find_package for Kokkos
add_subdirectory(/path/to/ponni  ponni )
add_library(MyProject ${MY_SOURCES})  # or add_executable
target_link_libraries(MyProject ponni)
```

## Ahead-of-time PyTorch/ONNX generator

The experimental [Kokkos neural-network generator](src/generator/README.md) exports fixed-shape PyTorch MLP DAGs to
ONNX, validates and optimizes a framework-neutral IR, and generates allocation-free Kokkos C++ with batched
`DeviceSpace`, batch-tiled hierarchical team-neuron/scratch, inline `SArray`, and Kokkos-launched packed
`ponni::TwoHalf` interfaces for CUDA/HIP with baseline and optional user-specified accumulator policies. It also emits
a standalone launch-bounds and batch-tile autotuner. ONNX and Python are build-time tools only; generated inference
does not link an ML runtime.

The supported model class is a fixed-feature, inference-only vector DAG: dense MLPs, residual and branched networks,
feature concatenation, supported activations, feature normalization/probabilities/reductions, scalar or exact-shape
elementwise arithmetic, and typed Boolean comparisons, logical masks, and selection. One dynamic batch dimension is
allowed. Convolution/pooling, attention,
recurrent/control-flow models, dynamic hidden or sequence dimensions, multiple inputs/outputs, arbitrary
broadcasting, training behavior, quantization, and custom ONNX operators are not yet supported. The generator emits
four families: inline one-sample `SArray`, View batch, hierarchical team-neuron, and packed two-sample FP16. See the
generator documentation for the exact ONNX operator matrix, launch-bound tuning, and half2 accumulator choices `0`,
`2`, `4`, `8`, `16`, and `32`.

## Activation layers

PONNI provides the following activation layers:

- `Relu` (`ReLU`), `LeakyRelu` (`LeakyReLU`), `Elu` (`ELU`), and `Selu` (`SELU`)
- `Gelu` (`GELU`), `Silu` (`SiLU`), `Sigmoid`, and `Tanh`
- `Softmax`, `LogSoftmax`, and `Softplus`
- `HardSigmoid`, `HardSwish`, and `Mish`

Include `ponni.h` to make all activation layers available. Each activation is implemented in its own
`src/layers/ponni_<Activation>.h` header, following the same one-layer-per-header organization as the other PONNI
layers.

Every activation supports both dynamic `Kokkos::View` execution and fixed-size `SArray` execution. Activation
configuration is validated by the layer itself and can be serialized with `to_array()` and restored with
`from_array()`. Activation layers have no trainable parameters, so `get_num_trainable_parameters()` returns zero and
`get_trainable_parameters()` returns an empty view.

## Unit tests

Run all unit tests:

```bash
git clone git@github.com:mrnorman/ponni.git
cd ponni
git submodule update --init --checkout -- unit/externals/kokkos
cd unit/build
# Choose or create a machine profile
source machines/thatchroof/thatchroof_cpu_coverage.env
./cmakescript.sh
make -j8
ctest -V
```

Notes:

- All unit tests are registered through CTest.
- The core unit test covers every activation's host API, configuration serialization, fixed-size `SArray` path, and
  accelerator-capable `DeviceSpace` path.
- The performance benchmark executable is built, but it is not registered as a CTest test and is not run by `ctest`.
- The `keras_sequential`, `keras_resnet`, and `pytorch_resnet` unit tests require Python-generated HDF5 test data.

### Python environment for unit tests

During the **make phase** (not configure), unit test dependencies are prepared with `uv`:

- CMake finds Python 3.x with `find_package(Python3 ...)`.
- `uv` is resolved and installed locally in the build tree at `unit/build/uv_env` (no install into `~/.local`).
- The Python virtual environment is created in `unit/build/python_env` and installs packages used by tests and
  ahead-of-time model generation:
	- `torch`
	- `keras`
	- `tensorflow`
	- `tf2onnx`
	- `numpy`
	- `h5py`
	- `onnx`
	- `onnxruntime`
	- `onnxscript`
- Python scripts generate test HDF5 files in the build tree, and C++ tests consume those generated files.

This unit-test workflow keeps tooling and Python dependencies inside `ponni/unit/build` and does not place files in the user's `~/.local`.

This workflow is intended for Linux/macOS (Windows is not supported by this path).

## Coverage tests with gcov

To run tests with gcov instrumentation and print per-file coverage summaries for all files discovered under `ponni/src`:

```bash
cd unit/build
source machines/thatchroof/thatchroof_cpu_coverage.env
export PONNI_COVERAGE=ON
./cmakescript.sh
make -j8
# Optional but recommended before collecting fresh coverage:
find . -name '*.gcda' -delete
ctest -V
```

The coverage `ctest` run also executes `src_gcov_summary_test`, which prints per-file coverage summaries for files under `src/`.

The CTest run includes `src_gcov_summary_test`, which executes `unit/report_src_gcov_summary.sh` and reports:

- Number of source files found recursively under `ponni/src`
- Per-file gcov line coverage summaries
- Files missing gcov data

Coverage artifacts are written to:

- `unit/build/coverage/src_gcov_summary.txt`
- `unit/build/coverage/gcov_intermediate_all.txt`

You can also run the summary script directly:

```bash
cd unit/build
/bin/bash ../report_src_gcov_summary.sh "$PWD"
```
