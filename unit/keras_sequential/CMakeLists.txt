
add_executable(keras_sequential test_keras_sequential.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(keras_sequential)
target_link_libraries(keras_sequential ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(keras_sequential "${HDF5_LINK_FLAGS}")
endif()

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_sequential_test COMMAND ${REAL_MPI_COMMAND} ./keras_sequential ${CMAKE_CURRENT_SOURCE_DIR}/keras_sequential_data.h5)
else()
  add_test(NAME keras_sequential_test COMMAND ./keras_sequential ${CMAKE_CURRENT_SOURCE_DIR}/keras_sequential_data.h5)
endif()

