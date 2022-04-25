
add_library(pinni INTERFACE)
target_link_libraries(pinni INTERFACE yakl)
target_include_directories(pinni INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

