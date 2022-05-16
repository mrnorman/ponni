
add_executable(performance_benchmark performance_benchmark.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(performance_benchmark)
target_link_libraries(performance_benchmark ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(performance_benchmark "${HDF5_LINK_FLAGS}")
endif()

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME performance_benchmark_test COMMAND ${REAL_MPI_COMMAND} ./performance_benchmark)
else()
  add_test(NAME performance_benchmark_test COMMAND ./performance_benchmark)
endif()

