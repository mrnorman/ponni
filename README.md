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
