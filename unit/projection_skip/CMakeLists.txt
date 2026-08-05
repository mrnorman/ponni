add_executable(projection_skip test_projection_skip.cpp)
target_link_libraries(projection_skip ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(projection_skip PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

set(PROJECTION_SKIP_WEIGHTS ${CMAKE_CURRENT_BINARY_DIR}/projection_skip_generated.ponni)
add_custom_command(
  OUTPUT ${PROJECTION_SKIP_WEIGHTS}
  COMMAND ${PONNI_TEST_PYTHON_COMMAND}
          ${CMAKE_CURRENT_SOURCE_DIR}/generate_projection_skip_data.py ${PROJECTION_SKIP_WEIGHTS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_projection_skip_data.py
          ${CMAKE_SOURCE_DIR}/../src/generator/kokkos_nn/weights.py
  COMMENT "Generating projection skip test data with Python"
  VERBATIM
)

add_custom_target(projection_skip_data_gen DEPENDS ${PROJECTION_SKIP_WEIGHTS})
add_dependencies(projection_skip_data_gen ponni_python_env)
add_dependencies(projection_skip projection_skip_data_gen)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME projection_skip_test COMMAND ${REAL_MPI_COMMAND} ./projection_skip ${PROJECTION_SKIP_WEIGHTS})
else()
  add_test(NAME projection_skip_test COMMAND ./projection_skip ${PROJECTION_SKIP_WEIGHTS})
endif()
