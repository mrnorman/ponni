cmake_minimum_required(VERSION 3.0)
project(ponni_unit)
enable_testing()

enable_language(CXX)
enable_language(C)

option(PONNI_ENABLE_COVERAGE "Enable gcov coverage flags and coverage report targets" OFF)

if (WIN32)
	message(FATAL_ERROR "Python uv test-data workflow is supported only on Linux and macOS")
endif()

set(PONNI_PYTHON_VERSION "" CACHE STRING "Build-local Python major.minor version (auto-select when empty)")

# Keep downloaded Python tooling under the unit build tree. This makes the
# test workflow hermetic and prevents accidental use of user-site packages.
set(PONNI_UNIT_BUILD_DIR "${CMAKE_SOURCE_DIR}/build")
set(PONNI_UV_ENV_DIR "${PONNI_UNIT_BUILD_DIR}/uv_env")
set(PONNI_UV_PATH_FILE "${PONNI_UV_ENV_DIR}/uv_path.txt")

add_custom_command(
	OUTPUT ${PONNI_UV_PATH_FILE}
	COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/ensure_uv.sh ${PONNI_UV_PATH_FILE} ${PONNI_UV_ENV_DIR}
	DEPENDS ${CMAKE_SOURCE_DIR}/ensure_uv.sh
	COMMENT "Ensuring the PONNI-owned uv installation is available"
	VERBATIM
)

add_custom_target(ponni_uv_tool DEPENDS ${PONNI_UV_PATH_FILE})

# Coverage flags must be attached before any PONNI or Kokkos targets are added.
if (PONNI_ENABLE_COVERAGE)
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		add_compile_options(-O0 -g --coverage)
		add_link_options(--coverage)
	else()
		message(WARNING "PONNI_ENABLE_COVERAGE requested, but compiler is not GCC/Clang. Coverage flags were not applied.")
	endif()
endif()

# Build Kokkos and PONNI once; individual test directories only define fixtures
# and executables against these targets.
add_subdirectory($ENV{KOKKOS_HOME} ./kokkos)
add_subdirectory(.. ./ponni)

# Frameworks only export and validate backend-neutral ONNX. Generated C++ is
# tested with the configured Kokkos backend independently.
set(PONNI_TEST_FRAMEWORK_BACKEND CPU)

if (PONNI_PYTHON_VERSION STREQUAL "")
	set(PONNI_PYTHON_VERSION "3.11")
elseif (PONNI_PYTHON_VERSION VERSION_LESS "3.11")
	message(FATAL_ERROR "PONNI generator tests require Python 3.11 or newer for current CPU ONNX Runtime")
endif()

set(PONNI_TEST_PYTHON_INSTALL_DIR "${PONNI_UV_ENV_DIR}/python_install")
set(PONNI_TEST_PYTHON_ENV_DIR "${PONNI_UNIT_BUILD_DIR}/python_cpu_env")
set(PONNI_TEST_PYTHON_EXECUTABLE "${PONNI_TEST_PYTHON_ENV_DIR}/bin/python")

add_custom_command(
	OUTPUT ${PONNI_TEST_PYTHON_ENV_DIR}/.ready
	COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/ensure_python_env.sh ${PONNI_UV_PATH_FILE} ${PONNI_PYTHON_VERSION}
	        ${PONNI_TEST_PYTHON_INSTALL_DIR} ${PONNI_TEST_PYTHON_ENV_DIR} ${PONNI_TEST_FRAMEWORK_BACKEND}
	DEPENDS ${PONNI_UV_PATH_FILE}
	DEPENDS ${CMAKE_SOURCE_DIR}/ensure_python_env.sh
	COMMENT "Ensuring the PONNI-owned ${PONNI_TEST_FRAMEWORK_BACKEND} Python environment is available"
	VERBATIM
)

add_custom_target(ponni_python_env DEPENDS ${PONNI_TEST_PYTHON_ENV_DIR}/.ready)
add_dependencies(ponni_python_env ponni_uv_tool)

set(PONNI_TEST_PYTHON_COMMAND ${CMAKE_COMMAND} -E env
    PYTHONHOME= PYTHONNOUSERSITE=1 PYTHONPATH=${CMAKE_SOURCE_DIR}/../src/generator
    "PYTHONWARNINGS=ignore::FutureWarning:copyreg,ignore::FutureWarning:keras.src.export.tf2onnx_lib")
if (NOT Kokkos_ENABLE_CUDA)
	list(APPEND PONNI_TEST_PYTHON_COMMAND
	     CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3)
endif()
list(APPEND PONNI_TEST_PYTHON_COMMAND ${PONNI_TEST_PYTHON_EXECUTABLE})

# Test suites. Keep the generator last because it consumes the Python target
# and produces the largest set of generated build artifacts.
add_subdirectory(keras_sequential)
add_subdirectory(performance_benchmark)
add_subdirectory(pytorch_resnet)
add_subdirectory(keras_resnet)
add_subdirectory(projection_skip)
add_subdirectory(core core_unit_tests)
add_subdirectory(generator)

# Coverage is an explicit post-build workflow: run all tests first, then collect
# compiler data into a stable text report and optional gcovr HTML output.
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
