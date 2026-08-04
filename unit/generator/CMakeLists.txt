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
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py --output-dir ${PONNI_GENERATOR_BUILD_DIR}
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_framework_models.py
          --output-dir ${PONNI_GENERATOR_BUILD_DIR}
  DEPENDS ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_models.py
          ${PONNI_GENERATOR_SOURCE_DIR}/examples/export_framework_models.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/export.py
          ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/framework_export.py
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
set(PONNI_GENERATED_OUTPUTS
    ${PONNI_MLP_DIR}/MlpModel.hpp
    ${PONNI_MLP_DIR}/MlpModel_autotune.cpp
    ${PONNI_MLP_DIR}/weights.bin
    ${PONNI_MLP_DIR}/optimization_report.json
    ${PONNI_RESIDUAL_DIR}/ResidualModel.hpp
    ${PONNI_RESIDUAL_DIR}/ResidualModel_autotune.cpp
    ${PONNI_RESIDUAL_DIR}/weights.bin
    ${PONNI_RESIDUAL_DIR}/optimization_report.json
    ${PONNI_DEEP10_DIR}/Deep10Model.hpp
    ${PONNI_DEEP10_DIR}/Deep10Model_autotune.cpp
    ${PONNI_DEEP10_DIR}/weights.bin
    ${PONNI_DEEP10_DIR}/optimization_report.json
    ${PONNI_RESNET10_DIR}/ResNet10Model.hpp
    ${PONNI_RESNET10_DIR}/ResNet10Model_autotune.cpp
    ${PONNI_RESNET10_DIR}/weights.bin
    ${PONNI_RESNET10_DIR}/optimization_report.json
    ${PONNI_DENSENET_DIR}/DenseNetModel.hpp
    ${PONNI_DENSENET_DIR}/DenseNetModel_autotune.cpp
    ${PONNI_DENSENET_DIR}/weights.bin
    ${PONNI_DENSENET_DIR}/optimization_report.json
    ${PONNI_BRANCHING_DIR}/BranchingModel.hpp
    ${PONNI_BRANCHING_DIR}/BranchingModel_autotune.cpp
    ${PONNI_BRANCHING_DIR}/weights.bin
    ${PONNI_BRANCHING_DIR}/optimization_report.json)
list(APPEND PONNI_GENERATED_OUTPUTS
    ${PONNI_OPERATOR_ZOO_DIR}/OperatorZooModel.hpp
    ${PONNI_OPERATOR_ZOO_DIR}/OperatorZooModel_autotune.cpp
    ${PONNI_OPERATOR_ZOO_DIR}/weights.bin
    ${PONNI_OPERATOR_ZOO_DIR}/optimization_report.json
    ${PONNI_KERAS_DIR}/KerasModel.hpp
    ${PONNI_KERAS_DIR}/KerasModel_autotune.cpp
    ${PONNI_KERAS_DIR}/weights.bin
    ${PONNI_KERAS_DIR}/optimization_report.json
    ${PONNI_KERAS_NORMALIZATION_DIR}/KerasNormalizationModel.hpp
    ${PONNI_KERAS_NORMALIZATION_DIR}/KerasNormalizationModel_autotune.cpp
    ${PONNI_KERAS_NORMALIZATION_DIR}/weights.bin
    ${PONNI_KERAS_NORMALIZATION_DIR}/optimization_report.json
    ${PONNI_TENSORFLOW_DIR}/TensorFlowModel.hpp
    ${PONNI_TENSORFLOW_DIR}/TensorFlowModel_autotune.cpp
    ${PONNI_TENSORFLOW_DIR}/weights.bin
    ${PONNI_TENSORFLOW_DIR}/optimization_report.json)

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
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/keras_mlp.onnx --output-dir ${PONNI_KERAS_DIR}
          --strategy auto --model-name KerasModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/keras_normalization.onnx
          --output-dir ${PONNI_KERAS_NORMALIZATION_DIR}
          --strategy auto --model-name KerasNormalizationModel
  COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
          ${PONNI_GENERATOR_BUILD_DIR}/tensorflow_residual.onnx --output-dir ${PONNI_TENSORFLOW_DIR}
          --strategy auto --model-name TensorFlowModel
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

add_executable(generator_autotuner ${PONNI_MLP_DIR}/MlpModel_autotune.cpp)
add_dependencies(generator_autotuner generator_model_generation)
target_include_directories(generator_autotuner PRIVATE ${PONNI_MLP_DIR})
target_link_libraries(generator_autotuner ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
target_compile_options(generator_autotuner PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)

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
  NAME generator_autotuner_test
  COMMAND generator_autotuner 100 ${PONNI_MLP_DIR}/weights.bin)

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
          --operator-zoo ${PONNI_OPERATOR_ZOO_DIR})
set_tests_properties(generator_structure_test PROPERTIES DEPENDS generator_integration_test)

if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
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
         ${PONNI_WIDTH_DIR}/Width${PONNI_WIDTH}Model_autotune.cpp
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
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/planner.py
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
  add_test(
    NAME generator_gpu_scale_test
    COMMAND bash ${PONNI_GENERATOR_SOURCE_DIR}/tests/run_gpu_scale_performance.sh
            $<TARGET_FILE:generator_gpu_scale> ${PONNI_WIDTH_WEIGHT_ARGUMENTS})
  set_tests_properties(generator_gpu_scale_test PROPERTIES LABELS "gpu;performance")

  set(PONNI_ARCHITECTURE_NAMES
      SeqW16D2 SeqW32D2 SeqW64D2 SeqW128D2 SeqW32D8 SeqW64D8
      ResidualW32D4 ResidualW64D4 ResidualW32D8 ResidualW64D8 ResidualW128D8
      LongSkipW32D8 LongSkipW64D8 Branch4W32D2 Branch4W64D2)
  set(PONNI_ARCHITECTURE_ROOT ${PONNI_GENERATOR_BUILD_DIR}/batch_team_architectures)
  set(PONNI_ARCHITECTURE_MODEL_DIR ${PONNI_ARCHITECTURE_ROOT}/models)
  set(PONNI_ARCHITECTURE_MODELS)
  set(PONNI_ARCHITECTURE_GENERATED_OUTPUTS)
  set(PONNI_ARCHITECTURE_INCLUDE_DIRS)
  set(PONNI_ARCHITECTURE_COMPILE_COMMANDS)
  foreach(PONNI_ARCHITECTURE_NAME IN LISTS PONNI_ARCHITECTURE_NAMES)
    set(PONNI_ARCHITECTURE_MODEL
        ${PONNI_ARCHITECTURE_MODEL_DIR}/${PONNI_ARCHITECTURE_NAME}.onnx)
    set(PONNI_ARCHITECTURE_DIR
        ${PONNI_ARCHITECTURE_ROOT}/generated/${PONNI_ARCHITECTURE_NAME})
    list(APPEND PONNI_ARCHITECTURE_MODELS ${PONNI_ARCHITECTURE_MODEL})
    list(APPEND PONNI_ARCHITECTURE_GENERATED_OUTPUTS
         ${PONNI_ARCHITECTURE_DIR}/${PONNI_ARCHITECTURE_NAME}.hpp
         ${PONNI_ARCHITECTURE_DIR}/weights.bin
         ${PONNI_ARCHITECTURE_DIR}/optimization_report.json)
    list(APPEND PONNI_ARCHITECTURE_INCLUDE_DIRS ${PONNI_ARCHITECTURE_DIR})
    list(APPEND PONNI_ARCHITECTURE_COMPILE_COMMANDS
      COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE} -m kokkos_nn compile
              ${PONNI_ARCHITECTURE_MODEL} --output-dir ${PONNI_ARCHITECTURE_DIR}
              --strategy batch-team --model-name ${PONNI_ARCHITECTURE_NAME}
              --streaming-recompute-threshold 0)
  endforeach()

  add_custom_command(
    OUTPUT ${PONNI_ARCHITECTURE_MODELS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PONNI_ARCHITECTURE_MODEL_DIR}
    COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
            ${PONNI_GENERATOR_SOURCE_DIR}/examples/create_batch_team_architecture_benchmarks.py
            --output-dir ${PONNI_ARCHITECTURE_MODEL_DIR}
    DEPENDS ${PONNI_GENERATOR_SOURCE_DIR}/examples/create_batch_team_architecture_benchmarks.py
    COMMENT "Creating deterministic ONNX batch-team architecture models"
    VERBATIM)

  add_custom_command(
    OUTPUT ${PONNI_ARCHITECTURE_GENERATED_OUTPUTS}
    ${PONNI_ARCHITECTURE_COMPILE_COMMANDS}
    DEPENDS ${PONNI_ARCHITECTURE_MODELS}
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/compiler.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/emitter.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/importer.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/ir.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/passes.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/planner.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/scheduler.py
            ${PONNI_GENERATOR_SOURCE_DIR}/kokkos_nn/weights.py
    COMMENT "Compiling batch-team architecture models to generated Kokkos C++"
    VERBATIM)
  add_custom_target(generator_batch_team_architecture_generation
                    DEPENDS ${PONNI_ARCHITECTURE_GENERATED_OUTPUTS})
  add_dependencies(generator_batch_team_architecture_generation ponni_python_env)

  add_executable(
    generator_batch_team_architecture_experiment
    ${PONNI_GENERATOR_SOURCE_DIR}/tests/benchmark_batch_team_architectures.cpp)
  add_dependencies(
    generator_batch_team_architecture_experiment
    generator_batch_team_architecture_generation)
  target_include_directories(
    generator_batch_team_architecture_experiment PRIVATE ${PONNI_ARCHITECTURE_INCLUDE_DIRS})
  target_link_libraries(
    generator_batch_team_architecture_experiment ponni Kokkos::kokkos ${PONNI_LINK_FLAGS})
  target_compile_options(
    generator_batch_team_architecture_experiment
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PONNI_CXX_FLAGS}>)
  add_test(
    NAME generator_batch_team_architecture_test
    COMMAND ${PONNI_GENERATOR_ENV} ${PONNI_TEST_PYTHON_EXECUTABLE}
            ${PONNI_GENERATOR_SOURCE_DIR}/tests/run_batch_team_architecture_experiment.py
            --executable $<TARGET_FILE:generator_batch_team_architecture_experiment>
            --root ${PONNI_ARCHITECTURE_ROOT})
  set_tests_properties(
    generator_batch_team_architecture_test PROPERTIES LABELS "gpu;performance")
endif()
