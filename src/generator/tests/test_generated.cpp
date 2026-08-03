#include "MlpModel.hpp"
#include "ResidualModel.hpp"
#include "Deep10Model.hpp"
#include "ResNet10Model.hpp"
#include "DenseNetModel.hpp"
#include "BranchingModel.hpp"
#include "OperatorZooModel.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct ReferenceCase {
  int batch_size;
  std::vector<float> inputs;
  std::vector<float> outputs;
};

struct ReferenceData {
  int num_inputs;
  int num_outputs;
  std::vector<ReferenceCase> cases;
};

ReferenceData load_reference(std::string const & path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open reference file: " + path);
  int num_cases;
  ReferenceData data;
  stream >> num_cases >> data.num_inputs >> data.num_outputs;
  for (int icase = 0; icase < num_cases; icase++) {
    ReferenceCase reference;
    stream >> reference.batch_size;
    reference.inputs.resize(data.num_inputs * reference.batch_size);
    reference.outputs.resize(data.num_outputs * reference.batch_size);
    for (float & value : reference.inputs) stream >> value;
    for (float & value : reference.outputs) stream >> value;
    data.cases.push_back(std::move(reference));
  }
  if (!stream) throw std::runtime_error("malformed reference file: " + path);
  return data;
}

template <class Model>
bool check_model(std::string const & weight_path, std::string const & reference_path, std::string const & label,
                 float & maximum_error) {
  Model model;
  std::string error;
  if (!model.load_weights(weight_path, &error)) {
    std::cerr << label << " weight loading failed: " << error << std::endl;
    return false;
  }
  ReferenceData const reference = load_reference(reference_path);
  if (reference.num_inputs != Model::num_inputs || reference.num_outputs != Model::num_outputs) {
    std::cerr << label << " reference dimensions do not match the generated model" << std::endl;
    return false;
  }
  bool passed = true;
  for (ReferenceCase const & test : reference.cases) {
    typename Model::InputView inputs("generator_inputs", Model::num_inputs, test.batch_size);
    typename Model::OutputView batch_outputs("generator_batch_outputs", Model::num_outputs, test.batch_size);
    typename Model::OutputView hierarchical_outputs("generator_hierarchical_outputs", Model::num_outputs,
                                                    test.batch_size);
    typename Model::OutputView hierarchical_tile1_outputs("generator_hierarchical_tile1_outputs", Model::num_outputs,
                                                          test.batch_size);
    typename Model::OutputView inline_outputs("generator_inline_outputs", Model::num_outputs, test.batch_size);
    typename Model::OutputView half2_outputs("generator_half2_outputs", Model::num_outputs, test.batch_size);
    typename Model::OutputView half2_heuristic_outputs("generator_half2_heuristic_outputs", Model::num_outputs,
                                                       test.batch_size);
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
    typename Model::OutputView tensorcore_outputs("generator_tensorcore_outputs", Model::num_outputs, test.batch_size);
#endif
    auto inputs_host = Kokkos::create_mirror_view(inputs);
    for (int i = 0; i < Model::num_inputs; i++) {
      for (int ibatch = 0; ibatch < test.batch_size; ibatch++) {
        inputs_host(i,ibatch) = test.inputs[i * test.batch_size + ibatch];
      }
    }
    Kokkos::deep_copy(inputs, inputs_host);
    model.infer_batch(inputs, batch_outputs);
    model.infer_batch_hierarchical(inputs, hierarchical_outputs);
    model.infer_batch_hierarchical(inputs, hierarchical_tile1_outputs, 1);
    model.infer_batch_half2(inputs, half2_outputs);
    model.infer_batch_half2_heuristic(inputs, half2_heuristic_outputs);
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
    if constexpr (Model::tensorcore_eligible) model.infer_batch_tensorcore(inputs, tensorcore_outputs);
#endif

    auto const device_model = model;
    Kokkos::parallel_for("GeneratedModel::embedded_infer_one", test.batch_size, KOKKOS_LAMBDA(int ibatch) {
      ponni::SArray<float,Model::num_inputs> sample_inputs;
      ponni::SArray<float,Model::num_outputs> sample_outputs;
      for (int i = 0; i < Model::num_inputs; i++) sample_inputs(i) = inputs(i,ibatch);
      device_model.infer_one(sample_inputs, sample_outputs);
      for (int i = 0; i < Model::num_outputs; i++) inline_outputs(i,ibatch) = sample_outputs(i);
    });

    auto batch_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch_outputs);
    auto hierarchical_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hierarchical_outputs);
    auto hierarchical_tile1_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hierarchical_tile1_outputs);
    auto inline_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inline_outputs);
    auto half2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), half2_outputs);
    auto half2_heuristic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), half2_heuristic_outputs);
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
    auto tensorcore_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tensorcore_outputs);
#endif
    for (int i = 0; i < Model::num_outputs; i++) {
      for (int ibatch = 0; ibatch < test.batch_size; ibatch++) {
        float const expected = test.outputs[i * test.batch_size + ibatch];
        float const batch_error = std::abs(batch_host(i,ibatch) - expected);
        float const hierarchical_error = std::abs(hierarchical_host(i,ibatch) - expected);
        float const hierarchical_tile1_error = std::abs(hierarchical_tile1_host(i,ibatch) - expected);
        float const inline_error = std::abs(inline_host(i,ibatch) - expected);
        float const half2_error = std::abs(half2_host(i,ibatch) - expected);
        float const half2_heuristic_error = std::abs(half2_heuristic_host(i,ibatch) - expected);
        maximum_error = std::max(maximum_error, std::max(batch_error, std::max(hierarchical_error, inline_error)));
        maximum_error = std::max(maximum_error, hierarchical_tile1_error);
        maximum_error = std::max(maximum_error, half2_error);
        maximum_error = std::max(maximum_error, half2_heuristic_error);
        if (batch_error > 2.e-5f || hierarchical_error > 2.e-5f || hierarchical_tile1_error > 2.e-5f ||
            inline_error > 2.e-5f) {
          std::cerr << label << " mismatch at batch size " << test.batch_size << ", output " << i
                    << ", sample " << ibatch << ": expected=" << expected
                    << ", batch=" << batch_host(i,ibatch)
                    << ", hierarchical=" << hierarchical_host(i,ibatch)
                    << ", hierarchical_tile1=" << hierarchical_tile1_host(i,ibatch)
                    << ", inline=" << inline_host(i,ibatch) << std::endl;
          passed = false;
        }
        if (half2_error > 3.e-2f || !std::isfinite(half2_error)) {
          std::cerr << label << " half2 mismatch at batch size " << test.batch_size << ", output " << i
                    << ", sample " << ibatch << ": expected=" << expected
                    << ", half2=" << half2_host(i,ibatch) << ", error=" << half2_error << std::endl;
          passed = false;
        }
        if (half2_heuristic_error > 3.e-2f || !std::isfinite(half2_heuristic_error)) {
          std::cerr << label << " heuristic half2 mismatch at batch size " << test.batch_size
                    << ", output " << i << ", sample " << ibatch << ": expected=" << expected
                    << ", half2=" << half2_heuristic_host(i,ibatch)
                    << ", error=" << half2_heuristic_error << std::endl;
          passed = false;
        }
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_AMPERE)
        if constexpr (Model::tensorcore_eligible) {
          float const tensorcore_error = std::abs(tensorcore_host(i,ibatch) - expected);
          maximum_error = std::max(maximum_error, tensorcore_error);
          if (tensorcore_error > 2.e-3f || !std::isfinite(tensorcore_error)) {
            std::cerr << label << " Tensor Core mismatch at batch size " << test.batch_size
                      << ", output " << i << ", sample " << ibatch
                      << ": expected=" << expected << ", tensorcore=" << tensorcore_host(i,ibatch)
                      << ", error=" << tensorcore_error << std::endl;
            passed = false;
          }
        }
#endif
      }
    }
  }
  return passed;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 15) {
    std::cerr << "Usage: " << argv[0]
              << " mlp_weights mlp_reference residual_weights residual_reference"
              << " deep10_weights deep10_reference resnet10_weights resnet10_reference"
              << " densenet_weights densenet_reference branching_weights branching_reference"
              << " operator_zoo_weights operator_zoo_reference" << std::endl;
    return 2;
  }
  Kokkos::initialize(argc, argv);
  ponni::init_device_pool(128ULL * 1024ULL * 1024ULL);
  bool passed = true;
  float maximum_error = 0;
  {
    using Mlp = ponni::generated::MlpModel<float>;
    using Residual = ponni::generated::ResidualModel<float>;
    using Deep10 = ponni::generated::Deep10Model<float>;
    using ResNet10 = ponni::generated::ResNet10Model<float>;
    using DenseNet = ponni::generated::DenseNetModel<float>;
    using Branching = ponni::generated::BranchingModel<float>;
    using OperatorZoo = ponni::generated::OperatorZooModel<float>;
    passed = check_model<Mlp>(argv[1], argv[2], "MLP", maximum_error) && passed;
    passed = check_model<Residual>(argv[3], argv[4], "residual", maximum_error) && passed;
    passed = check_model<Deep10>(argv[5], argv[6], "depth-10 MLP", maximum_error) && passed;
    passed = check_model<ResNet10>(argv[7], argv[8], "depth-10 ResNet", maximum_error) && passed;
    passed = check_model<DenseNet>(argv[9], argv[10], "DenseNet", maximum_error) && passed;
    passed = check_model<Branching>(argv[11], argv[12], "branching DAG", maximum_error) && passed;
    passed = check_model<OperatorZoo>(argv[13], argv[14], "operator zoo", maximum_error) && passed;
  }
  ponni::finalize_device_pool();
  Kokkos::finalize();
  std::cout << "generator integration maximum absolute error: " << maximum_error << std::endl;
  return passed ? 0 : 1;
}
