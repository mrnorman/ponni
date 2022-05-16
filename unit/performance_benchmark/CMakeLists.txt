
add_executable(performance_benchmark performance_benchmark.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(performance_benchmark)
target_link_libraries(performance_benchmark ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(performance_benchmark "${HDF5_LINK_FLAGS}")
endif()

