# Unit testing

Workflow to to run the unit tests:

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
- CPU Python dependencies are installed into that venv: `torch`, `keras`, `numpy`, `h5py`.
- Generator scripts produce HDF5 files in each test's build directory.

This workflow keeps uv and Python dependencies inside `ponni/unit/build` and does not install files under `~/.local`.

No Python package installation is performed during CMake configure.

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