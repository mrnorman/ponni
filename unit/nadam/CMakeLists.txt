
add_executable(nadam test_nadam.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(nadam)
target_link_libraries(nadam ponni)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME nadam_test COMMAND ${REAL_MPI_COMMAND} ./nadam)
else()
  add_test(NAME nadam_test COMMAND ./nadam)
endif()

