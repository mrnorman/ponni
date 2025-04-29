
add_executable(keras_sequential test_keras_sequential.cpp)
target_link_libraries(keras_sequential ponni yakl ${HDF5_LIBRARIES} ${PONNI_LINK_FLAGS})
target_compile_options(keras_sequential PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME keras_sequential_test COMMAND ${REAL_MPI_COMMAND} ./keras_sequential ${CMAKE_CURRENT_SOURCE_DIR}/keras_sequential_data.h5)
else()
  add_test(NAME keras_sequential_test COMMAND ./keras_sequential ${CMAKE_CURRENT_SOURCE_DIR}/keras_sequential_data.h5)
endif()

