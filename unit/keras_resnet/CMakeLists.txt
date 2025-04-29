
add_executable(keras_resnet test_keras_resnet.cpp)
target_link_libraries(keras_resnet ponni yakl ${HDF5_LIBRARIES} ${PONNI_LINK_FLAGS})
target_compile_options(keras_resnet PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_resnet_test COMMAND ${REAL_MPI_COMMAND} ./keras_resnet ${CMAKE_CURRENT_SOURCE_DIR}/keras_resnet_data.h5)
else()
  add_test(NAME keras_resnet_test COMMAND ./keras_resnet ${CMAKE_CURRENT_SOURCE_DIR}/keras_resnet_data.h5)
endif()

