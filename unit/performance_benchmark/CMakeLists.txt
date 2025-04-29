
add_executable(performance_benchmark performance_benchmark.cpp)
target_link_libraries(performance_benchmark ponni yakl ${HDF5_LIBRARIES} ${PONNI_LINK_FLAGS})
target_compile_options(performance_benchmark PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

