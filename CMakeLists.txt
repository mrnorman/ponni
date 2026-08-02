cmake_minimum_required(VERSION 3.0)
project(ponni)

add_library(ponni STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/ponni_DeviceSpace.cpp)
target_link_libraries(ponni PUBLIC Kokkos::kokkos)
target_include_directories(ponni PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(ponni PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/layers)
target_include_directories(ponni PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
target_include_directories(ponni PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/initializers)
