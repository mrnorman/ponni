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

To run the unit tests:
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
ctest
```
