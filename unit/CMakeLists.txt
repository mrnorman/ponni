cmake_minimum_required(VERSION 3.0)
project(ponni_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

# YAKL library
add_subdirectory($ENV{KOKKOS_HOME} ./kokkos )
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory($ENV{YAKL_HOME}   ./yakl   )
add_subdirectory(..                ./ponni  )

include(FindHDF5)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

# Unit test directories
add_subdirectory(keras_sequential)
add_subdirectory(performance_benchmark)
add_subdirectory(pytorch_resnet)
add_subdirectory(keras_resnet)

