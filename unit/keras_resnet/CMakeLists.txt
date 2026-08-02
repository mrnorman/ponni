
add_executable(keras_resnet test_keras_resnet.cpp)
target_link_libraries(keras_resnet ponni Kokkos::kokkos ${HDF5_LIBRARIES} ${PONNI_LINK_FLAGS})
target_compile_options(keras_resnet PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

set(KERAS_RESNET_H5 ${CMAKE_CURRENT_BINARY_DIR}/keras_resnet_generated.h5)
add_custom_command(
  OUTPUT ${KERAS_RESNET_H5}
  COMMAND ${PONNI_TEST_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_keras_resnet_data.py ${KERAS_RESNET_H5}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_keras_resnet_data.py
  COMMENT "Generating Keras ResNet test data with Python"
  VERBATIM
)

add_custom_target(keras_resnet_data_gen DEPENDS ${KERAS_RESNET_H5})
add_dependencies(keras_resnet_data_gen ponni_python_env)
add_dependencies(keras_resnet keras_resnet_data_gen)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_resnet_test COMMAND ${REAL_MPI_COMMAND} ./keras_resnet ${KERAS_RESNET_H5})
else()
  add_test(NAME keras_resnet_test COMMAND ./keras_resnet ${KERAS_RESNET_H5})
endif()

