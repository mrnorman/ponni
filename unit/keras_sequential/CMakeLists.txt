
add_executable(keras_sequential test_keras_sequential.cpp)
target_link_libraries(keras_sequential ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(keras_sequential PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

set(KERAS_SEQUENTIAL_WEIGHTS ${CMAKE_CURRENT_BINARY_DIR}/keras_sequential_generated.ponni)
add_custom_command(
  OUTPUT ${KERAS_SEQUENTIAL_WEIGHTS}
  COMMAND ${PONNI_TEST_PYTHON_COMMAND}
          ${CMAKE_CURRENT_SOURCE_DIR}/generate_keras_sequential_data.py ${KERAS_SEQUENTIAL_WEIGHTS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_keras_sequential_data.py
          ${CMAKE_SOURCE_DIR}/../src/generator/kokkos_nn/weights.py
  COMMENT "Generating Keras sequential test data with Python"
  VERBATIM
)

add_custom_target(keras_sequential_data_gen DEPENDS ${KERAS_SEQUENTIAL_WEIGHTS})
add_dependencies(keras_sequential_data_gen ponni_python_env)
add_dependencies(keras_sequential keras_sequential_data_gen)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_sequential_test COMMAND ${REAL_MPI_COMMAND} ./keras_sequential ${KERAS_SEQUENTIAL_WEIGHTS})
else()
  add_test(NAME keras_sequential_test COMMAND ./keras_sequential ${KERAS_SEQUENTIAL_WEIGHTS})
endif()
