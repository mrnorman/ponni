
add_library(ponni INTERFACE)
target_link_libraries(ponni INTERFACE yakl)
target_include_directories(ponni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

