
add_executable(keras_resnet test_keras_resnet.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(keras_resnet)
target_link_libraries(keras_resnet ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(keras_resnet "${HDF5_LINK_FLAGS}")
endif()

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_resnet_test COMMAND ${REAL_MPI_COMMAND} ./keras_resnet ${CMAKE_CURRENT_SOURCE_DIR}/keras_resnet_data.h5)
else()
  add_test(NAME keras_resnet_test COMMAND ./keras_resnet ${CMAKE_CURRENT_SOURCE_DIR}/keras_resnet_data.h5)
endif()

