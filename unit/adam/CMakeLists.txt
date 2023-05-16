
add_executable(adam test_adam.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(adam)
target_link_libraries(adam ponni)

# if (DEFINED MPI_COMMAND)
#   separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
#   add_test(NAME adam_test COMMAND ${REAL_MPI_COMMAND} ./adam)
# else()
#   add_test(NAME adam_test COMMAND ./adam)
# endif()
# 
