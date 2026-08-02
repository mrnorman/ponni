cmake_minimum_required(VERSION 3.0)
project(ponni_unit)
enable_testing()

enable_language(CXX)
enable_language(C)

option(PONNI_ENABLE_COVERAGE "Enable gcov coverage flags and coverage report targets" OFF)

if (WIN32)
	message(FATAL_ERROR "Python uv test-data workflow is supported only on Linux and macOS")
endif()

find_package(Python3 3 REQUIRED COMPONENTS Interpreter)
find_program(UV_EXECUTABLE uv)

set(PONNI_UV_PATH_FILE "${CMAKE_BINARY_DIR}/uv_path.txt")
set(PONNI_UV_ENV_DIR "${CMAKE_BINARY_DIR}/uv_env")

add_custom_command(
	OUTPUT ${PONNI_UV_PATH_FILE}
	COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/ensure_uv.sh ${PONNI_UV_PATH_FILE} ${Python3_EXECUTABLE} ${PONNI_UV_ENV_DIR} "${UV_EXECUTABLE}"
	DEPENDS ${CMAKE_SOURCE_DIR}/ensure_uv.sh
	COMMENT "Ensuring uv is available (installing in build/uv_env if needed)"
	VERBATIM
)

add_custom_target(ponni_uv_tool DEPENDS ${PONNI_UV_PATH_FILE})

set(PONNI_TEST_PYTHON_ENV_DIR "${CMAKE_BINARY_DIR}/python_env")
set(PONNI_TEST_PYTHON_EXECUTABLE "${PONNI_TEST_PYTHON_ENV_DIR}/bin/python")

add_custom_command(
	OUTPUT ${PONNI_TEST_PYTHON_ENV_DIR}/.ready
	COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/ensure_python_env.sh ${PONNI_UV_PATH_FILE} ${Python3_EXECUTABLE} ${PONNI_TEST_PYTHON_ENV_DIR}
	DEPENDS ${PONNI_UV_PATH_FILE}
	DEPENDS ${CMAKE_SOURCE_DIR}/ensure_python_env.sh
	COMMAND ${CMAKE_COMMAND} -E touch ${PONNI_TEST_PYTHON_ENV_DIR}/.ready
	COMMENT "Ensuring Python test environment (uv + CPU torch + keras)"
	VERBATIM
)

add_custom_target(ponni_python_env DEPENDS ${PONNI_TEST_PYTHON_ENV_DIR}/.ready)
add_dependencies(ponni_python_env ponni_uv_tool)

if (PONNI_ENABLE_COVERAGE)
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		add_compile_options(-O0 -g --coverage)
		add_link_options(--coverage)
	else()
		message(WARNING "PONNI_ENABLE_COVERAGE requested, but compiler is not GCC/Clang. Coverage flags were not applied.")
	endif()
endif()

add_subdirectory($ENV{KOKKOS_HOME} ./kokkos )
add_subdirectory(..                ./ponni  )

include(FindHDF5)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

# Unit test directories
add_subdirectory(keras_sequential)
add_subdirectory(performance_benchmark)
add_subdirectory(pytorch_resnet)
add_subdirectory(keras_resnet)
add_subdirectory(projection_skip)
add_subdirectory(core core_unit_tests)

add_test(NAME src_gcov_summary_test
		 COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/report_src_gcov_summary.sh ${CMAKE_BINARY_DIR})
set_tests_properties(src_gcov_summary_test PROPERTIES
					 DEPENDS "keras_sequential_test;pytorch_resnet_test;keras_resnet_test;projection_skip_test;core_unit_test")

if (PONNI_ENABLE_COVERAGE)
	find_program(GCOV_EXECUTABLE gcov)
	find_program(GCOVR_EXECUTABLE gcovr)

	if (GCOV_EXECUTABLE)
		add_custom_target(coverage
			COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
			COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
			COMMAND ${CMAKE_COMMAND}
					-DGCOV_EXECUTABLE=${GCOV_EXECUTABLE}
					-DCOVERAGE_BUILD_DIR=${CMAKE_BINARY_DIR}
					-P ${CMAKE_SOURCE_DIR}/run_gcov.cmake
			COMMAND ${CMAKE_COMMAND} -E echo "gcov report written to ${CMAKE_BINARY_DIR}/coverage/gcov.txt"
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
			COMMENT "Running unit tests and generating gcov coverage report"
		)

		if (GCOVR_EXECUTABLE)
			add_custom_command(TARGET coverage POST_BUILD
				COMMAND ${GCOVR_EXECUTABLE}
						--root ${CMAKE_SOURCE_DIR}/..
						--object-directory ${CMAKE_BINARY_DIR}
						--gcov-executable ${GCOV_EXECUTABLE}
						--filter ${CMAKE_SOURCE_DIR}/../src
						--exclude ${CMAKE_SOURCE_DIR}/../unit/externals
						--print-summary
						--txt ${CMAKE_BINARY_DIR}/coverage/coverage.txt
						--html-details ${CMAKE_BINARY_DIR}/coverage/index.html
				WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
				COMMENT "gcovr detected: generating summary text/html reports"
			)
		endif()
	else()
		add_custom_target(coverage
			COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
			COMMAND ${CMAKE_COMMAND} -E echo "gcov not found: tests ran with coverage flags, but report generation was skipped."
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
			COMMENT "Running unit tests with coverage instrumentation"
		)
	endif()
endif()

