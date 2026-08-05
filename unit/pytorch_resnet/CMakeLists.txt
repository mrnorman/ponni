
add_executable(pytorch_resnet test_pytorch_resnet.cpp)
target_link_libraries(pytorch_resnet ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(pytorch_resnet PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

set(PYTORCH_RESNET_WEIGHTS ${CMAKE_CURRENT_BINARY_DIR}/pytorch_resnet_generated.ponni)
add_custom_command(
  OUTPUT ${PYTORCH_RESNET_WEIGHTS}
  COMMAND ${PONNI_TEST_PYTHON_COMMAND}
          ${CMAKE_CURRENT_SOURCE_DIR}/generate_pytorch_resnet_data.py ${PYTORCH_RESNET_WEIGHTS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_pytorch_resnet_data.py
          ${CMAKE_SOURCE_DIR}/../src/generator/kokkos_nn/weights.py
  COMMENT "Generating PyTorch ResNet test data with Python"
  VERBATIM
)

add_custom_target(pytorch_resnet_data_gen DEPENDS ${PYTORCH_RESNET_WEIGHTS})
add_dependencies(pytorch_resnet_data_gen ponni_python_env)
add_dependencies(pytorch_resnet pytorch_resnet_data_gen)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME pytorch_resnet_test COMMAND ${REAL_MPI_COMMAND} ./pytorch_resnet ${PYTORCH_RESNET_WEIGHTS})
else()
  add_test(NAME pytorch_resnet_test COMMAND ./pytorch_resnet ${PYTORCH_RESNET_WEIGHTS})
endif()
