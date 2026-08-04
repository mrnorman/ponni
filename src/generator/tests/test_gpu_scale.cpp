#include "Width4Model.hpp"
#include "Width8Model.hpp"
#include "Width16Model.hpp"
#include "Width32Model.hpp"
#include "Width64Model.hpp"
#include "Width128Model.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

template <class Function>
double time_inference(Function const & function, int iterations) {
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int iteration = 0; iteration < iterations; iteration++) function();
  Kokkos::fence();
  return timer.seconds() / iterations;
}

template <class Model>
bool infer_batch_team(Model const & model, typename Model::InputView const & inputs,
                      typename Model::OutputView const & outputs, int team_size) {
  switch (team_size) {
    case 64: return model.try_infer_batch_team_64(inputs, outputs);
    case 128: return model.try_infer_batch_team_128(inputs, outputs);
    case 256: return model.try_infer_batch_team_256(inputs, outputs);
    case 512: return model.try_infer_batch_team_512(inputs, outputs);
    case 1024: return model.try_infer_batch_team_1024(inputs, outputs);
    default: return false;
  }
}

template <class Model>
bool benchmark_model(char const * weight_path, int hidden_width) {
  Model model;
  std::string error;
  if (!model.load_weights(weight_path, &error)) {
    std::cerr << error << std::endl;
    return false;
  }

  bool passed = true;
  for (int batch_size : {10000, 100000, 1000000}) {
    int const iterations = batch_size == 10000 ? 500 : (batch_size == 100000 ? 100 : 10);
    typename Model::InputView inputs("gpu_scale_inputs", Model::num_inputs, batch_size);
    typename Model::OutputView batch_outputs("gpu_scale_batch_outputs", Model::num_outputs, batch_size);
    typename Model::OutputView sarray_outputs("gpu_scale_sarray_outputs", Model::num_outputs, batch_size);
    typename Model::OutputView batch_team_outputs("gpu_scale_batch_team_outputs", Model::num_outputs, batch_size);
    typename Model::OutputView half2_outputs("gpu_scale_half2_outputs", Model::num_outputs, batch_size);
    Kokkos::parallel_for(
        "generator_gpu_initialize_inputs",
        Kokkos::RangePolicy<typename Model::execution_space>(0, Model::num_inputs * batch_size),
        KOKKOS_LAMBDA(int linear) {
          int const ibatch = linear % batch_size;
          int const iinput = linear / batch_size;
          inputs(iinput,ibatch) = 0.25f + 0.01f * iinput + 0.001f * (ibatch % 17);
        });

    model.infer_batch(inputs, batch_outputs);
    Kokkos::fence();
    double const batch_seconds = time_inference(
        [&]() { model.infer_batch(inputs, batch_outputs); }, iterations);
    std::size_t const output_elements = static_cast<std::size_t>(Model::num_outputs) * batch_size;
    Model const device_model = model;
    auto const infer_sarray = [&]() {
      Kokkos::parallel_for("generator_gpu_sarray", batch_size, KOKKOS_LAMBDA(int ibatch) {
        ponni::SArray<float,Model::num_inputs> sample_inputs;
        ponni::SArray<float,Model::num_outputs> sample_outputs;
        for (int i = 0; i < Model::num_inputs; i++) sample_inputs(i) = inputs(i,ibatch);
        device_model.infer_one(sample_inputs, sample_outputs);
        for (int i = 0; i < Model::num_outputs; i++) sarray_outputs(i,ibatch) = sample_outputs(i);
      });
    };
    infer_sarray();
    Kokkos::fence();
    double const sarray_seconds = time_inference(infer_sarray, iterations);
    auto const infer_half2 = [&](int variant) {
      if (variant == 0) {
        model.infer_batch_half2(inputs, half2_outputs);
      } else {
        model.infer_batch_half2_explicit(inputs, half2_outputs);
      }
    };
    int constexpr half2_variant_count = 2;
    char const * const half2_variant_names[half2_variant_count] = {"none", "explicit"};
    double half2_seconds[half2_variant_count] = {};
    float half2_errors[half2_variant_count] = {};
    double best_half2_seconds = std::numeric_limits<double>::max();
    int best_half2_variant = 0;
    float best_half2_error = std::numeric_limits<float>::max();
    int most_accurate_half2_variant = 0;
    for (int ivariant = 0; ivariant < half2_variant_count; ivariant++) {
      infer_half2(ivariant);
      Kokkos::fence();
      half2_seconds[ivariant] = time_inference(
          [&]() { infer_half2(ivariant); }, iterations);
      Kokkos::parallel_reduce(
          "generator_gpu_half2_accumulator_error",
          Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
          KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
            int const ibatch = static_cast<int>(linear % batch_size);
            int const ioutput = static_cast<int>(linear / batch_size);
            float const error_value = Kokkos::abs(
                batch_outputs(ioutput,ibatch) - half2_outputs(ioutput,ibatch));
            if (error_value > error_max) error_max = error_value;
          },
          Kokkos::Max<float>(half2_errors[ivariant]));
      if (half2_seconds[ivariant] < best_half2_seconds) {
        best_half2_seconds = half2_seconds[ivariant];
        best_half2_variant = ivariant;
      }
      if (half2_errors[ivariant] < best_half2_error) {
        best_half2_error = half2_errors[ivariant];
        most_accurate_half2_variant = ivariant;
      }
      std::cout << "generator_gpu_half2_policy width=" << hidden_width
                << " batch=" << batch_size
                << " policy=" << half2_variant_names[ivariant]
                << " half2_ms=" << half2_seconds[ivariant] * 1.e3
                << " max_abs_difference=" << half2_errors[ivariant] << std::endl;
      if (half2_errors[ivariant] > 5.e-2f || !std::isfinite(half2_errors[ivariant])) passed = false;
    }
    double team_64_seconds = 0;
    double best_batch_team_seconds = std::numeric_limits<double>::max();
    int best_batch_team_size = 0;
    float worst_error = 0;
    float worst_sarray_error = 0;
    Kokkos::parallel_reduce(
        "generator_gpu_sarray_error",
        Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
        KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
          int const ibatch = static_cast<int>(linear % batch_size);
          int const ioutput = static_cast<int>(linear / batch_size);
          float const error_value = Kokkos::abs(
              batch_outputs(ioutput,ibatch) - sarray_outputs(ioutput,ibatch));
          if (error_value > error_max) error_max = error_value;
        },
        Kokkos::Max<float>(worst_sarray_error));
    for (int team_size : {64, 128, 256, 512, 1024}) {
      if (!infer_batch_team(model, inputs, batch_team_outputs, team_size)) {
        std::cout << "generator_gpu_batch_team_unsupported width=" << hidden_width
                  << " batch=" << batch_size
                  << " team_size=" << team_size << std::endl;
        continue;
      }
      Kokkos::fence();
      double const batch_team_seconds = time_inference(
          [&]() { (void) infer_batch_team(model, inputs, batch_team_outputs, team_size); }, iterations);
      if (team_size == 64) team_64_seconds = batch_team_seconds;
      if (batch_team_seconds < best_batch_team_seconds) {
        best_batch_team_seconds = batch_team_seconds;
        best_batch_team_size = team_size;
      }

      float maximum_error = 0;
      Kokkos::parallel_reduce(
          "generator_gpu_batch_team_error",
          Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
          KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
            int const ibatch = static_cast<int>(linear % batch_size);
            int const ioutput = static_cast<int>(linear / batch_size);
            float const error_value = Kokkos::abs(
                batch_outputs(ioutput,ibatch) - batch_team_outputs(ioutput,ibatch));
            if (error_value > error_max) error_max = error_value;
          },
          Kokkos::Max<float>(maximum_error));
      if (maximum_error > worst_error) worst_error = maximum_error;

      std::cout << "generator_gpu_batch_team width=" << hidden_width
                << " batch=" << batch_size
                << " team_size=" << team_size
                << " batch_only_ms=" << batch_seconds * 1.e3
                << " sarray_ms=" << sarray_seconds * 1.e3
                << " batch_team_ms=" << batch_team_seconds * 1.e3
                << " speedup=" << batch_seconds / batch_team_seconds
                << " max_abs_difference=" << maximum_error
                << " sarray_max_abs_difference=" << worst_sarray_error << std::endl;
      if (maximum_error > 2.e-6f || !std::isfinite(maximum_error)) passed = false;
    }
    if (best_batch_team_size == 0 || team_64_seconds == 0) passed = false;
    if (worst_sarray_error > 2.e-6f || !std::isfinite(worst_sarray_error)) passed = false;
    std::cout << "generator_gpu_summary width=" << hidden_width
              << " batch=" << batch_size
              << " sarray_ms=" << sarray_seconds * 1.e3
              << " view_batch_ms=" << batch_seconds * 1.e3
              << " batch_team_64_ms=" << team_64_seconds * 1.e3
              << " batch_team_best_ms=" << best_batch_team_seconds * 1.e3
              << " batch_team_best_size=" << best_batch_team_size
              << " half2_ms=" << half2_seconds[0] * 1.e3
              << " half2_max_abs_difference=" << half2_errors[0]
              << " half2_best_ms=" << best_half2_seconds * 1.e3
              << " half2_best_policy=" << half2_variant_names[best_half2_variant]
              << " half2_best_error=" << best_half2_error
              << " half2_most_accurate_policy=" << half2_variant_names[most_accurate_half2_variant]
              << " max_abs_difference=" << worst_error
              << " sarray_max_abs_difference=" << worst_sarray_error << std::endl;
  }
  return passed;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " width4.bin width8.bin width16.bin width32.bin width64.bin width128.bin"
              << std::endl;
    return 2;
  }
  Kokkos::initialize(argc, argv);
  ponni::init_device_pool(1024ULL * 1024ULL * 1024ULL);
  bool passed = true;
  {
    passed = benchmark_model<ponni::generated::Width4Model<float>>(argv[1], 4) && passed;
    passed = benchmark_model<ponni::generated::Width8Model<float>>(argv[2], 8) && passed;
    passed = benchmark_model<ponni::generated::Width16Model<float>>(argv[3], 16) && passed;
    passed = benchmark_model<ponni::generated::Width32Model<float>>(argv[4], 32) && passed;
    passed = benchmark_model<ponni::generated::Width64Model<float>>(argv[5], 64) && passed;
    passed = benchmark_model<ponni::generated::Width128Model<float>>(argv[6], 128) && passed;
  }
  ponni::finalize_device_pool();
  Kokkos::finalize();
  return passed ? 0 : 1;
}
