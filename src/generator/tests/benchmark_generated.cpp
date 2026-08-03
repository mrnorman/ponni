#include "MlpModel.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " weights.bin" << std::endl;
    return 2;
  }
  Kokkos::initialize(argc, argv);
  ponni::init_device_pool(128ULL * 1024ULL * 1024ULL);
  {
    using Model = ponni::generated::MlpModel<float>;
    Model model;
    std::string error;
    auto const cold_start = std::chrono::steady_clock::now();
    if (!model.load_weights(argv[1], &error)) {
      std::cerr << error << std::endl;
      return 1;
    }
    Kokkos::fence();
    auto const cold_end = std::chrono::steady_clock::now();
    std::cout << "weight load milliseconds: "
              << std::chrono::duration<double,std::milli>(cold_end - cold_start).count() << std::endl;
    for (int batch_size : {1, 32, 1024, 4099}) {
      Model::InputView inputs("benchmark_inputs", Model::num_inputs, batch_size);
      Model::OutputView batch_outputs("benchmark_batch_outputs", Model::num_outputs, batch_size);
      Model::OutputView hierarchical_outputs("benchmark_hierarchical_outputs", Model::num_outputs, batch_size);
      Kokkos::deep_copy(inputs, 0.25f);
      model.infer_batch(inputs, batch_outputs);
      model.infer_batch_hierarchical(inputs, hierarchical_outputs);
      Kokkos::fence();
      int constexpr iterations = 100;
      auto const batch_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < iterations; iteration++) model.infer_batch(inputs, batch_outputs);
      Kokkos::fence();
      auto const batch_end = std::chrono::steady_clock::now();
      auto const hierarchical_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < iterations; iteration++) {
        model.infer_batch_hierarchical(inputs, hierarchical_outputs);
      }
      Kokkos::fence();
      auto const hierarchical_end = std::chrono::steady_clock::now();
      double const batch_seconds = std::chrono::duration<double>(batch_end - batch_start).count();
      double const hierarchical_seconds =
          std::chrono::duration<double>(hierarchical_end - hierarchical_start).count();
      std::cout << "batch=" << batch_size << " batch_only_latency_us=" << batch_seconds * 1.e6 / iterations
                << " hierarchical_latency_us=" << hierarchical_seconds * 1.e6 / iterations
                << " batch_only_samples_per_second=" << iterations * batch_size / batch_seconds
                << " hierarchical_samples_per_second=" << iterations * batch_size / hierarchical_seconds
                << std::endl;

      auto const device_model = model;
      auto const embedded_start = std::chrono::steady_clock::now();
      Kokkos::parallel_for("embedded_nn_benchmark", batch_size, KOKKOS_LAMBDA(int ibatch) {
        ponni::SArray<float,Model::num_inputs> sample_inputs;
        ponni::SArray<float,Model::num_outputs> sample_outputs;
        for (int i = 0; i < Model::num_inputs; i++) sample_inputs(i) = inputs(i,ibatch);
        device_model.infer_one(sample_inputs, sample_outputs);
        for (int i = 0; i < Model::num_outputs; i++) batch_outputs(i,ibatch) = sample_outputs(i);
      });
      Kokkos::fence();
      auto const embedded_end = std::chrono::steady_clock::now();
      std::cout << "batch=" << batch_size << " embedded_infer_one_us="
                << std::chrono::duration<double,std::micro>(embedded_end - embedded_start).count() << std::endl;
    }
  }
  ponni::finalize_device_pool();
  Kokkos::finalize();
  return 0;
}
