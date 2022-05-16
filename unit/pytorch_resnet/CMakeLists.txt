
add_executable(pytorch_resnet test_pytorch_resnet.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(pytorch_resnet)
target_link_libraries(pytorch_resnet ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(pytorch_resnet "${HDF5_LINK_FLAGS}")
endif()

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME pytorch_resnet_test COMMAND ${REAL_MPI_COMMAND} ./pytorch_resnet ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_resnet_data.h5)
else()
  add_test(NAME pytorch_resnet_test COMMAND ./pytorch_resnet ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_resnet_data.h5)
endif()

