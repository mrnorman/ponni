
add_executable(pytorch_resnet test_pytorch_resnet.cpp)
target_link_libraries(pytorch_resnet ponni yakl ${HDF5_LIBRARIES} ${PONNI_LINK_FLAGS})
target_compile_options(pytorch_resnet PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME pytorch_resnet_test COMMAND ${REAL_MPI_COMMAND} ./pytorch_resnet ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_resnet_data.h5)
else()
  add_test(NAME pytorch_resnet_test COMMAND ./pytorch_resnet ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_resnet_data.h5)
endif()

