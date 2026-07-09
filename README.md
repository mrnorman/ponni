# PONNI: Portable Online Neural Network Inferencing
### Efficient in-loop neural network inferencing made easy in C++

Author: Matt Norman, Oak Ridge National Laboratory, https://mrnorman.github.io

PONNI provides a convenient way to build an efficient, portable Neural Network inference model in C++ with minimal syntax and full disclosure of exactly how the model is running on an accelerator device. It is built on the [YAKL is A Kokkos Layer (YAKL)](https://github.com/mrnorman/YAKL) portable C++ library and runs out of the box on Nvidia, AMD, and Intel GPUs as well as CPUs with or without OpenMP 3.5 threading using the Kokkos library.

## Using PONNI in your CMake project
PONNI currently uses the YAKL is A Kokkos Layer library, though there are efforts underway to make Kokkos the only dependency. YAKL is largely a header-only wrapper to the Kokkos library with very little source to compile, and therefore it currently uses the `add_subdirectory` CMake approach. Feel free to add YAKL as a git submodule to your project alongside PONNI.

First, in your `CMakeLists.txt`, you need to add the Kokkos library either through `find_package` or through `add_subdirectory`:
https://kokkos.org/kokkos-core-wiki/get-started/integrating-kokkos-into-your-cmake-project.html

Once Kokkos has been added, you need to add the YAKL library with `add_subdirectory(${YAKL_HOME} ${YAKL_BIN})`, where you can replace YAKL_HOME and YAKL_BIN with the location you have placed the YAKL git clone / module and the location you would like the build to go, respectively.
https://github.com/mrnorman/YAKL/#example-compilation-approach

Finally, you can add PONNI to your project with `add_subdirectory(${PONNI_HOME} ${PONNI_BIN})`, again with those CMake variables being replaced with the location you have placed the PONNI git clone / module and the location you would like the build to go, respectively.

YAKL automatically links itself to Kokkos, and PONNI automatically links itself to YAKL as library dependencies. Once you've added PONNI to the project, you need to link your targets to PONNI with `target_link_libraries`.

```bash
git clone --branch 5.1.0 git@github.com:kokkos/kokkos.git
git clone git@github.com:mrnorman/YAKL.git
git clone git@github.com:mrnorman/ponni.git
```

```CMake
cmake_minimum_required(VERSION 3.22)
project(MyProject)
add_subdirectory(/path/to/kokkos kokkos) # or use find_package for Kokkos
add_subdirectory(/path/to/YAKL   yakl  )
add_subdirectory(/path/to/ponni  ponni )

add_library(MyProject ${MY_SOURCES})  # or add_executable
target_link_libraries(MyProject ponni)
```

Because PONNI is header-only, it will use the compiler flags you set in CMake for your source files that use PONNI. The same is true for YAKL as well. The Kokkos build will repond to the CMake variables you set before calling add_library in the example above. YAKL responds to the Kokkos debug flags that get set through Kokkos' CMake variables as well.
