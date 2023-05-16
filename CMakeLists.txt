cmake_minimum_required(VERSION 3.0)
project(ponni)

add_library(ponni INTERFACE)
target_link_libraries(ponni INTERFACE yakl)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/layers)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/trainers)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/initializers)

