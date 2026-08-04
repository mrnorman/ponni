set(PONNI_GENERATOR_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../src/generator)
set(PONNI_GENERATOR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(PONNI_GENERATOR_ENV ${CMAKE_COMMAND} -E env
    PYTHONHOME= PYTHONNOUSERSITE=1 PYTHONPATH=${PONNI_GENERATOR_SOURCE_DIR}
    "PYTHONWARNINGS=ignore::FutureWarning:copyreg,ignore::FutureWarning:keras.src.export.tf2onnx_lib")
if (NOT Kokkos_ENABLE_CUDA)
  list(APPEND PONNI_GENERATOR_ENV
       CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3)
endif()

set(PONNI_GENERATOR_MODELS
    ${PONNI_GENERATOR_BUILD_DIR}/mlp.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/residual.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/deep10.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/resnet10.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/densenet.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/branching.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/workspace_levels.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/keras_mlp.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_residual.onnx
    ${PONNI_GENERATOR_BUILD_DIR}/mlp_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/residual_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/deep10_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/resnet10_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/densenet_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/branching_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/workspace_levels_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/keras_mlp_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_residual_reference.txt
    ${PONNI_GENERATOR_BUILD_DIR}/export_report.json
    ${PONNI_GENERATOR_BUILD_DIR}/framework_export_report.json)

add_custom_command(
  OUTPUT ${PONNI_GENERATOR_MODELS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PONNI_GENERATOR_BUILD_DIR}
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py
          --output-dir ${PONNI_GENERATOR_BUILD_DIR} --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_framework_models.py
          --output-dir ${PONNI_GENERATOR_BUILD_DIR} --quiet
  DEPENDS ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_framework_models.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/export.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/framework_export.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/onnx_reference.py
  COMMENT "Exporting PONNI generator PyTorch, Keras, and TensorFlow examples to ONNX"
  VERBATIM)

set(PONNI_MLP_DIR ${PONNI_GENERATOR_BUILD_DIR}/mlp_generated)
set(PONNI_RESIDUAL_DIR ${PONNI_GENERATOR_BUILD_DIR}/residual_generated)
set(PONNI_DEEP10_DIR ${PONNI_GENERATOR_BUILD_DIR}/deep10_generated)
set(PONNI_RESNET10_DIR ${PONNI_GENERATOR_BUILD_DIR}/resnet10_generated)
set(PONNI_DENSENET_DIR ${PONNI_GENERATOR_BUILD_DIR}/densenet_generated)
set(PONNI_BRANCHING_DIR ${PONNI_GENERATOR_BUILD_DIR}/branching_generated)
set(PONNI_OPERATOR_ZOO_DIR ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_generated)
set(PONNI_KERAS_DIR ${PONNI_GENERATOR_BUILD_DIR}/keras_generated)
set(PONNI_KERAS_NORMALIZATION_DIR ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization_generated)
set(PONNI_TENSORFLOW_DIR ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_generated)
set(PONNI_WORKSPACE_LEVEL_ROOT ${PONNI_GENERATOR_BUILD_DIR}/workspace_levels_generated)
set(PONNI_WORKSPACE_LEVEL_OUTPUTS)
set(PONNI_WORKSPACE_LEVEL_INCLUDE_DIRS)
set(PONNI_WORKSPACE_LEVEL_COMPILE_COMMANDS)
foreach(PONNI_WORKSPACE_LEVEL RANGE 1 5)
  set(PONNI_WORKSPACE_LEVEL_DIR ${PONNI_WORKSPACE_LEVEL_ROOT}/level_${PONNI_WORKSPACE_LEVEL})
  list(APPEND PONNI_WORKSPACE_LEVEL_OUTPUTS
       ${PONNI_WORKSPACE_LEVEL_DIR}/WorkspaceLevel${PONNI_WORKSPACE_LEVEL}Model.hpp
       ${PONNI_WORKSPACE_LEVEL_DIR}/weights.bin
       ${PONNI_WORKSPACE_LEVEL_DIR}/optimization_report.json)
  list(APPEND PONNI_WORKSPACE_LEVEL_INCLUDE_DIRS ${PONNI_WORKSPACE_LEVEL_DIR})
  list(APPEND PONNI_WORKSPACE_LEVEL_COMPILE_COMMANDS
    COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
            ${PONNI_GENERATOR_BUILD_DIR}/workspace_levels.onnx
            --output-dir ${PONNI_WORKSPACE_LEVEL_DIR}
            --model-name WorkspaceLevel${PONNI_WORKSPACE_LEVEL}Model
            --workspace-reduction-aggressiveness ${PONNI_WORKSPACE_LEVEL} --quiet)
endforeach()
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
    ${PONNI_OPERATOR_ZOO_DIR}/optimization_report.json
    ${PONNI_KERAS_DIR}/KerasModel.hpp
    ${PONNI_KERAS_DIR}/weights.bin
    ${PONNI_KERAS_DIR}/optimization_report.json
    ${PONNI_KERAS_NORMALIZATION_DIR}/KerasNormalizationModel.hpp
    ${PONNI_KERAS_NORMALIZATION_DIR}/weights.bin
    ${PONNI_KERAS_NORMALIZATION_DIR}/optimization_report.json
    ${PONNI_TENSORFLOW_DIR}/TensorFlowModel.hpp
    ${PONNI_TENSORFLOW_DIR}/weights.bin
    ${PONNI_TENSORFLOW_DIR}/optimization_report.json
    ${PONNI_WORKSPACE_LEVEL_OUTPUTS})

add_custom_command(
  OUTPUT ${PONNI_GENERATED_OUTPUTS}
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/mlp.onnx --output-dir ${PONNI_MLP_DIR}
          --model-name MlpModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/residual.onnx --output-dir ${PONNI_RESIDUAL_DIR}
          --model-name ResidualModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/deep10.onnx --output-dir ${PONNI_DEEP10_DIR}
          --model-name Deep10Model --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/resnet10.onnx --output-dir ${PONNI_RESNET10_DIR}
          --model-name ResNet10Model --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/densenet.onnx --output-dir ${PONNI_DENSENET_DIR}
          --model-name DenseNetModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/branching.onnx --output-dir ${PONNI_BRANCHING_DIR}
          --model-name BranchingModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo.onnx --output-dir ${PONNI_OPERATOR_ZOO_DIR}
          --model-name OperatorZooModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/keras_mlp.onnx --output-dir ${PONNI_KERAS_DIR}
          --model-name KerasModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization.onnx
          --output-dir ${PONNI_KERAS_NORMALIZATION_DIR}
          --model-name KerasNormalizationModel --quiet
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_residual.onnx --output-dir ${PONNI_TENSORFLOW_DIR}
          --model-name TensorFlowModel --quiet
  ${PONNI_WORKSPACE_LEVEL_COMPILE_COMMANDS}
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
  ${PONNI_DENSENET_DIR} ${PONNI_BRANCHING_DIR} ${PONNI_OPERATOR_ZOO_DIR}
  ${PONNI_WORKSPACE_LEVEL_INCLUDE_DIRS})
target_link_libraries(generator_integration ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(generator_integration PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

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
          ${PONNI_OPERATOR_ZOO_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/operator_zoo_reference.txt
          ${PONNI_WORKSPACE_LEVEL_ROOT}/level_1/weights.bin
          ${PONNI_WORKSPACE_LEVEL_ROOT}/level_2/weights.bin
          ${PONNI_WORKSPACE_LEVEL_ROOT}/level_3/weights.bin
          ${PONNI_WORKSPACE_LEVEL_ROOT}/level_4/weights.bin
          ${PONNI_WORKSPACE_LEVEL_ROOT}/level_5/weights.bin
          ${PONNI_GENERATOR_BUILD_DIR}/workspace_levels_reference.txt)

add_executable(generator_framework_integration ${PONNI_GENERATOR_SOURCE_DIR}/tests/test_framework_generated.cpp)
add_dependencies(generator_framework_integration generator_model_generation)
target_include_directories(generator_framework_integration PRIVATE
  ${PONNI_KERAS_DIR} ${PONNI_KERAS_NORMALIZATION_DIR} ${PONNI_TENSORFLOW_DIR})
target_link_libraries(generator_framework_integration ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(generator_framework_integration PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)
add_test(
  NAME generator_framework_integration_test
  COMMAND generator_framework_integration
          ${PONNI_KERAS_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/keras_mlp_reference.txt
          ${PONNI_KERAS_NORMALIZATION_DIR}/weights.bin
          ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization_reference.txt
          ${PONNI_TENSORFLOW_DIR}/weights.bin ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_residual_reference.txt)
add_test(
  NAME generator_structure_test
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
          ${PONNI_GENERATOR_SOURCE_DIR}/tests/check_functionality_generation.py
          --deep ${PONNI_DEEP10_DIR} --resnet ${PONNI_RESNET10_DIR}
          --densenet ${PONNI_DENSENET_DIR} --branching ${PONNI_BRANCHING_DIR}
          --operator-zoo ${PONNI_OPERATOR_ZOO_DIR}
          --workspace-levels ${PONNI_WORKSPACE_LEVEL_ROOT})
set_tests_properties(generator_structure_test PROPERTIES DEPENDS generator_integration_test)
