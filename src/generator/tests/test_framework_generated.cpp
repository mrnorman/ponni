#include "KerasModel.hpp"
#include "KerasNormalizationModel.hpp"
#include "TensorFlowModel.hpp"
#include "generator_test_utils.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using ponni::test::ReferenceCase;
using ponni::test::ReferenceData;
using ponni::test::load_reference;

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
    typename Model::InputView inputs("framework_inputs", Model::num_inputs, test.batch_size);
    typename Model::OutputView batch_outputs("framework_batch", Model::num_outputs, test.batch_size);
    typename Model::OutputView inline_outputs("framework_inline", Model::num_outputs, test.batch_size);
    typename Model::OutputView half2_outputs("framework_half2", Model::num_outputs, test.batch_size);
    // Framework references are host data; stage them through a host mirror.
    auto inputs_host = Kokkos::create_mirror_view(inputs);
    for (int i = 0; i < Model::num_inputs; i++) {
      for (int ibatch = 0; ibatch < test.batch_size; ibatch++) {
        inputs_host(i,ibatch) = test.inputs[i * test.batch_size + ibatch];
      }
    }
    Kokkos::deep_copy(inputs, inputs_host);
    model.infer_batch(inputs, batch_outputs);
    model.infer_batch_half2(inputs, half2_outputs);

    // Exercise infer_one in its intended setting: embedded in a device kernel.
    auto const device_model = model;
    Kokkos::parallel_for("FrameworkModel::embedded_infer_one", test.batch_size, KOKKOS_LAMBDA(int ibatch) {
      ponni::SArray<float,Model::num_inputs> sample_inputs;
      ponni::SArray<float,Model::num_outputs> sample_outputs;
      for (int i = 0; i < Model::num_inputs; i++) sample_inputs(i) = inputs(i,ibatch);
      device_model.infer_one(sample_inputs, sample_outputs);
      for (int i = 0; i < Model::num_outputs; i++) inline_outputs(i,ibatch) = sample_outputs(i);
    });

    auto batch_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch_outputs);
    auto inline_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inline_outputs);
    auto half2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), half2_outputs);
    for (int i = 0; i < Model::num_outputs; i++) {
      for (int ibatch = 0; ibatch < test.batch_size; ibatch++) {
        float const expected = test.outputs[i * test.batch_size + ibatch];
        float const batch_error = std::abs(batch_host(i,ibatch) - expected);
        float const inline_error = std::abs(inline_host(i,ibatch) - expected);
        float const half2_error = std::abs(half2_host(i,ibatch) - expected);
        maximum_error = std::max(maximum_error, std::max(batch_error, inline_error));
        maximum_error = std::max(maximum_error, half2_error);
        if (batch_error > 2.e-5f || inline_error > 2.e-5f ||
            half2_error > 3.e-2f || !std::isfinite(half2_error)) {
          std::cerr << label << " mismatch at batch " << test.batch_size << ", output " << i
                    << ", sample " << ibatch << ": expected=" << expected
                    << ", batch=" << batch_host(i,ibatch)
                    << ", inline=" << inline_host(i,ibatch) << ", half2=" << half2_host(i,ibatch) << std::endl;
          passed = false;
        }
      }
    }
  }
  return passed;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " keras_weights keras_reference keras_normalization_weights keras_normalization_reference"
              << " tensorflow_weights tensorflow_reference" << std::endl;
    return 2;
  }
  Kokkos::initialize(argc, argv);
  bool passed = true;
  float maximum_error = 0;
  {
    using Keras = ponni::generated::KerasModel<float>;
    using KerasNormalization = ponni::generated::KerasNormalizationModel<float>;
    using TensorFlow = ponni::generated::TensorFlowModel<float>;
    passed = check_model<Keras>(argv[1], argv[2], "Keras MLP", maximum_error) && passed;
    passed = check_model<KerasNormalization>(argv[3], argv[4], "Keras normalization", maximum_error) && passed;
    passed = check_model<TensorFlow>(argv[5], argv[6], "TensorFlow residual", maximum_error) && passed;
  }
  Kokkos::finalize();
  std::cout << "framework generator maximum absolute error: " << maximum_error << std::endl;
  return passed ? 0 : 1;
}
