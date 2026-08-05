if (NOT DEFINED COVERAGE_BUILD_DIR)
  message(FATAL_ERROR "COVERAGE_BUILD_DIR is required")
endif()

if (NOT DEFINED GCOV_EXECUTABLE)
  message(FATAL_ERROR "GCOV_EXECUTABLE is required")
endif()

file(MAKE_DIRECTORY "${COVERAGE_BUILD_DIR}/coverage")

execute_process(
  COMMAND /bin/bash -lc "cd '${COVERAGE_BUILD_DIR}' && find . -name '*.gcda' -print0 | xargs -0 -r '${GCOV_EXECUTABLE}' -pb -r > coverage/gcov.txt"
  RESULT_VARIABLE GCOV_RESULT
  OUTPUT_VARIABLE GCOV_STDOUT
  ERROR_VARIABLE GCOV_STDERR
)

if (NOT GCOV_RESULT EQUAL 0)
  message(FATAL_ERROR "gcov generation failed with code ${GCOV_RESULT}\n${GCOV_STDOUT}\n${GCOV_STDERR}")
endif()
