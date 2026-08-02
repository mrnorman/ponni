# PONNI: Portable Online Neural Network Inferencing
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

## Unit tests

Run all unit tests:

```bash
git clone git@github.com:mrnorman/ponni.git
cd ponni
git submodule update --init --checkout -- unit/build/externals/kokkos
cd unit/build
# Choose or create a machine profile
source machines/thatchroof/thatchroof_cpu_coverage.env
./cmakescript.sh
make -j8
ctest -V
```

Notes:

- All unit tests are registered through CTest.
- The performance benchmark executable is built, but it is not registered as a CTest test and is not run by `ctest`.
- The `keras_sequential`, `keras_resnet`, and `pytorch_resnet` unit tests require Python-generated HDF5 test data.

### Python environment for unit tests

During the **make phase** (not configure), unit test dependencies are prepared with `uv`:

- CMake finds Python 3.x with `find_package(Python3 ...)`.
- `uv` is resolved and installed locally in the build tree at `unit/build/uv_env` (no install into `~/.local`).
- The Python virtual environment is created in `unit/build/python_env` and installs CPU packages used by tests:
	- `torch`
	- `keras`
	- `numpy`
	- `h5py`
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
