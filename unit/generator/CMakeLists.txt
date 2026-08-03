set(PONNI_GENERATOR_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../src/generator)
set(PONNI_GENERATOR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(PONNI_GENERATOR_ENV ${CMAKE_COMMAND} -E env PYTHONPATH=${PONNI_GENERATOR_SOURCE_DIR})

set(PONNI_GENERATOR_MODELS
    ${PONNI_GENERATOR_BUILD_DIR}/mlp.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/residual.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/deep10.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/resnet10.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/densenet.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/branching.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/mlp_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/residual_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/deep10_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/resnet10_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/densenet_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/branching_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/export_report.json)

add_custom_command(
  OUTPUT ${PONNI_GENERATOR_MODELS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PONNI_GENERATOR_BUILD_DIR}
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py --output-dir ${PONNI_GENERATOR_BUILD_DIR}
  DEPENDS ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/export.py
  COMMENT "Exporting PONNI generator PyTorch examples to ONNX"
  VERBATIM)

set(PONNI_MLP_DIR ${PONNI_GENERATOR_BUILD_DIR}/mlp_generated)
set(PONNI_RESIDUAL_DIR ${PONNI_GENERATOR_BUILD_DIR}/residual_generated)
set(PONNI_DEEP10_DIR ${PONNI_GENERATOR_BUILD_DIR}/deep10_generated)
set(PONNI_RESNET10_DIR ${PONNI_GENERATOR_BUILD_DIR}/resnet10_generated)
set(PONNI_DENSENET_DIR ${PONNI_GENERATOR_BUILD_DIR}/densenet_generated)
set(PONNI_BRANCHING_DIR ${PONNI_GENERATOR_BUILD_DIR}/branching_generated)
set(PONNI_OPERATOR_ZOO_DIR ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_generated)
set(PONNI_GENERATED_OUTPUTS
    ${PONNI_MLP_DIR}/MlpModel.hpp
    ${PONNI_MLP_DIR}/weights.bin
    ${PONNI_MLP_DIR}/optimization_report.json
    ${PONNI_RESIDUAL_DIR}/ResidualModel.hpp
    ${PONNI_RESIDUAL_DIR}/weights.bin
    ${PONNI_RESIDUAL_DIR}/optimization_report.json
    ${PONNI_DEEP10_DIR}/Deep10Model.hpp
    ${PONNI_DEEP10_DIR}/weights.bin
    ${PONNI_DEEP10_DIR}/optimization_report.json
    ${PONNI_RESNET10_DIR}/ResNet10Model.hpp
    ${PONNI_RESNET10_DIR}/weights.bin
    ${PONNI_RESNET10_DIR}/optimization_report.json
    ${PONNI_DENSENET_DIR}/DenseNetModel.hpp
    ${PONNI_DENSENET_DIR}/weights.bin
    ${PONNI_DENSENET_DIR}/optimization_report.json
    ${PONNI_BRANCHING_DIR}/BranchingModel.hpp
    ${PONNI_BRANCHING_DIR}/weights.bin
    ${PONNI_BRANCHING_DIR}/optimization_report.json)
list(APPEND PONNI_GENERATED_OUTPUTS
    ${PONNI_OPERATOR_ZOO_DIR}/OperatorZooModel.hpp
    ${PONNI_OPERATOR_ZOO_DIR}/weights.bin
    ${PONNI_OPERATOR_ZOO_DIR}/optimization_report.json)

add_custom_command(
  OUTPUT ${PONNI_GENERATED_OUTPUTS}
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/mlp.onnx --output-dir ${PONNI_MLP_DIR}
          --strategy auto --model-name MlpModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/residual.onnx --output-dir ${PONNI_RESIDUAL_DIR}
          --strategy auto --model-name ResidualModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/deep10.onnx --output-dir ${PONNI_DEEP10_DIR}
          --strategy auto --model-name Deep10Model
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/resnet10.onnx --output-dir ${PONNI_RESNET10_DIR}
          --strategy auto --model-name ResNet10Model
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/densenet.onnx --output-dir ${PONNI_DENSENET_DIR}
          --strategy auto --model-name DenseNetModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/branching.onnx --output-dir ${PONNI_BRANCHING_DIR}
          --strategy auto --model-name BranchingModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo.onnx --output-dir ${PONNI_OPERATOR_ZOO_DIR}
          --strategy auto --model-name OperatorZooModel
  DEPENDS ${PONNI_GENERATOR_MODELS}
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/compiler.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/emitter.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/importer.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/interpreter.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/ir.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/passes.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/planner.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/scheduler.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/weights.py
  COMMENT "Compiling ONNX examples to generated Kokkos C++"
  VERBATIM)

add_custom_target(generator_model_generation DEPENDS ${PONNI_GENERATED_OUTPUTS})
add_dependencies(generator_model_generation ponni_python_env)

add_executable(generator_integration ${PONNI_GENERATOR_SOURCE_DIR}/tests/test_generated.cpp)
add_dependencies(generator_integration generator_model_generation)
target_include_directories(generator_integration PRIVATE
  ${PONNI_MLP_DIR} ${PONNI_RESIDUAL_DIR} ${PONNI_DEEP10_DIR} ${PONNI_RESNET10_DIR}
  ${PONNI_DENSENET_DIR} ${PONNI_BRANCHING_DIR} ${PONNI_OPERATOR_ZOO_DIR})
target_link_libraries(generator_integration ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(generator_integration PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

add_executable(generator_benchmark ${PONNI_GENERATOR_SOURCE_DIR}/tests/benchmark_generated.cpp)
add_dependencies(generator_benchmark generator_model_generation)
target_include_directories(generator_benchmark PRIVATE ${PONNI_MLP_DIR})
target_link_libraries(generator_benchmark ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(generator_benchmark PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

add_test(
  NAME generator_python_test
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m unittest discover
          -s ${PONNI_GENERATOR_SOURCE_DIR}/tests -p test_*.py -v)
add_test(
  NAME generator_integration_test
  COMMAND generator_integration
          ${PONNI_MLP_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/mlp_reference.txt
          ${PONNI_RESIDUAL_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/residual_reference.txt
          ${PONNI_DEEP10_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/deep10_reference.txt
          ${PONNI_RESNET10_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/resnet10_reference.txt
          ${PONNI_DENSENET_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/densenet_reference.txt
          ${PONNI_BRANCHING_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/branching_reference.txt
          ${PONNI_OPERATOR_ZOO_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_reference.txt)
add_test(
  NAME generator_structure_test
  COMMAND ${PONNI_TEST_PYTHON_EXECUTABLE} ${PONNI_GENERATOR_SOURCE_DIR}/tests/check_functionality_generation.py
          --deep ${PONNI_DEEP10_DIR} --resnet ${PONNI_RESNET10_DIR}
          --densenet ${PONNI_DENSENET_DIR} --branching ${PONNI_BRANCHING_DIR}
          --operator-zoo ${PONNI_OPERATOR_ZOO_DIR})
set_tests_properties(generator_structure_test PROPERTIES DEPENDS generator_integration_test)

if (Kokkos_ENABLE_CUDA)
  set(PONNI_WIDTHS 4 8 16 32 64 128)
  set(PONNI_WIDTH_MODELS)
  set(PONNI_WIDTH_GENERATED_OUTPUTS)
  set(PONNI_WIDTH_INCLUDE_DIRS)
  set(PONNI_WIDTH_WEIGHT_ARGUMENTS)
  foreach(PONNI_WIDTH IN LISTS PONNI_WIDTHS)
    list(APPEND PONNI_WIDTH_MODELS ${PONNI_GENERATOR_BUILD_DIR}/width_${PONNI_WIDTH}.onnx)
    set(PONNI_WIDTH_DIR ${PONNI_GENERATOR_BUILD_DIR}/width_${PONNI_WIDTH}_generated)
    list(APPEND PONNI_WIDTH_GENERATED_OUTPUTS
         ${PONNI_WIDTH_DIR}/Width${PONNI_WIDTH}Model.hpp
         ${PONNI_WIDTH_DIR}/weights.bin
         ${PONNI_WIDTH_DIR}/optimization_report.json)
    list(APPEND PONNI_WIDTH_INCLUDE_DIRS ${PONNI_WIDTH_DIR})
    list(APPEND PONNI_WIDTH_WEIGHT_ARGUMENTS ${PONNI_WIDTH_DIR}/weights.bin)
  endforeach()

  add_custom_command(
    OUTPUT ${PONNI_WIDTH_MODELS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PONNI_GENERATOR_BUILD_DIR}
    COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
            ${PONNI_GENERATOR_SOURCE_DIR}/examples/create_width_benchmarks.py
            --output-dir ${PONNI_GENERATOR_BUILD_DIR}
    DEPENDS ${PONNI_GENERATOR_SOURCE_DIR}/examples/create_width_benchmarks.py
    COMMENT "Creating deterministic ONNX width benchmark models"
    VERBATIM)

  set(PONNI_WIDTH_COMPILE_COMMANDS)
  foreach(PONNI_WIDTH IN LISTS PONNI_WIDTHS)
    set(PONNI_WIDTH_DIR ${PONNI_GENERATOR_BUILD_DIR}/width_${PONNI_WIDTH}_generated)
    list(APPEND PONNI_WIDTH_COMPILE_COMMANDS
      COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
              ${PONNI_GENERATOR_BUILD_DIR}/width_${PONNI_WIDTH}.onnx
              --output-dir ${PONNI_WIDTH_DIR} --strategy auto
              --model-name Width${PONNI_WIDTH}Model --half2-accumulators 4)
  endforeach()
  add_custom_command(
    OUTPUT ${PONNI_WIDTH_GENERATED_OUTPUTS}
    ${PONNI_WIDTH_COMPILE_COMMANDS}
    DEPENDS ${PONNI_WIDTH_MODELS}
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/compiler.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/emitter.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/scheduler.py
    COMMENT "Compiling width benchmark models to generated Kokkos C++"
    VERBATIM)
  add_custom_target(generator_width_model_generation DEPENDS ${PONNI_WIDTH_GENERATED_OUTPUTS})
  add_dependencies(generator_width_model_generation ponni_python_env)

  add_executable(generator_gpu_scale ${PONNI_GENERATOR_SOURCE_DIR}/tests/test_gpu_scale.cpp)
  add_dependencies(generator_gpu_scale generator_width_model_generation)
  target_include_directories(generator_gpu_scale PRIVATE ${PONNI_WIDTH_INCLUDE_DIRS})
  target_link_libraries(generator_gpu_scale ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
  target_compile_options(generator_gpu_scale PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)
  add_test(NAME generator_gpu_scale_test COMMAND generator_gpu_scale ${PONNI_WIDTH_WEIGHT_ARGUMENTS})
  set_tests_properties(generator_gpu_scale_test PROPERTIES LABELS "gpu;performance")
endif()
