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
    typename Model::OutputView tiled_outputs("gpu_scale_tiled_outputs", Model::num_outputs, batch_size);
    typename Model::OutputView tensorcore_outputs("gpu_scale_tensorcore_outputs", Model::num_outputs, batch_size);
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
      } else if (variant == 1) {
        model.infer_batch_half2_heuristic(inputs, half2_outputs);
      } else {
        model.infer_batch_half2_explicit(inputs, half2_outputs);
      }
    };
    int constexpr half2_variant_count = 3;
    char const * const half2_variant_names[half2_variant_count] = {"none", "heuristic", "explicit"};
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
    double tensorcore_warp_one_seconds = 0;
    double best_tensorcore_seconds = std::numeric_limits<double>::max();
    int best_tensorcore_warps = 0;
    for (int warps_per_block : {1, 2, 4, 8}) {
      if (warps_per_block > Model::maximum_tensorcore_warps_per_block) continue;
      model.infer_batch_tensorcore(inputs, tensorcore_outputs, warps_per_block);
      Kokkos::fence();
      double const tensorcore_seconds = time_inference(
          [&]() { model.infer_batch_tensorcore(inputs, tensorcore_outputs, warps_per_block); }, iterations);
      if (warps_per_block == 1) tensorcore_warp_one_seconds = tensorcore_seconds;
      if (tensorcore_seconds < best_tensorcore_seconds) {
        best_tensorcore_seconds = tensorcore_seconds;
        best_tensorcore_warps = warps_per_block;
      }
      std::cout << "generator_gpu_tensorcore width=" << hidden_width
                << " batch=" << batch_size
                << " warps_per_block=" << warps_per_block
                << " tensorcore_ms=" << tensorcore_seconds * 1.e3 << std::endl;
    }
    double tile_one_seconds = 0;
    double best_hierarchical_seconds = std::numeric_limits<double>::max();
    int best_hierarchical_tile = 0;
    float worst_error = 0;
    float worst_sarray_error = 0;
    float tensorcore_error = 0;
    Kokkos::parallel_reduce(
        "generator_gpu_tensorcore_error",
        Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
        KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
          int const ibatch = static_cast<int>(linear % batch_size);
          int const ioutput = static_cast<int>(linear / batch_size);
          float const error_value = Kokkos::abs(
              batch_outputs(ioutput,ibatch) - tensorcore_outputs(ioutput,ibatch));
          if (error_value > error_max) error_max = error_value;
        },
        Kokkos::Max<float>(tensorcore_error));
    if (tensorcore_error > 2.e-3f || !std::isfinite(tensorcore_error)) passed = false;
    for (int batch_tile : {1, 2, 4, 8, 16, 32}) {
      if (batch_tile > Model::maximum_hierarchical_batch_tile) continue;
      model.infer_batch_hierarchical(inputs, tiled_outputs, batch_tile);
      Kokkos::fence();
      double const tiled_seconds = time_inference(
          [&]() { model.infer_batch_hierarchical(inputs, tiled_outputs, batch_tile); }, iterations);
      if (batch_tile == 1) tile_one_seconds = tiled_seconds;
      if (tiled_seconds < best_hierarchical_seconds) {
        best_hierarchical_seconds = tiled_seconds;
        best_hierarchical_tile = batch_tile;
      }

      float maximum_error = 0;
      float maximum_sarray_error = 0;
      Kokkos::parallel_reduce(
          "generator_gpu_scale_error",
          Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
          KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
            int const ibatch = static_cast<int>(linear % batch_size);
            int const ioutput = static_cast<int>(linear / batch_size);
            float const error_value = Kokkos::abs(
                batch_outputs(ioutput,ibatch) - tiled_outputs(ioutput,ibatch));
            if (error_value > error_max) error_max = error_value;
          },
          Kokkos::Max<float>(maximum_error));
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
          Kokkos::Max<float>(maximum_sarray_error));
      if (maximum_error > worst_error) worst_error = maximum_error;
      if (maximum_sarray_error > worst_sarray_error) worst_sarray_error = maximum_sarray_error;

      std::cout << "generator_gpu_tile width=" << hidden_width
                << " batch=" << batch_size
                << " tile=" << batch_tile
                << " default_tile=" << Model::default_hierarchical_batch_tile
                << " batch_only_ms=" << batch_seconds * 1.e3
                << " sarray_ms=" << sarray_seconds * 1.e3
                << " tiled_ms=" << tiled_seconds * 1.e3
                << " speedup=" << batch_seconds / tiled_seconds
                << " max_abs_difference=" << maximum_error
                << " sarray_max_abs_difference=" << maximum_sarray_error << std::endl;
      if (maximum_error > 2.e-6f || maximum_sarray_error > 2.e-6f ||
          !std::isfinite(maximum_error) || !std::isfinite(maximum_sarray_error)) passed = false;
    }
    std::cout << "generator_gpu_summary width=" << hidden_width
              << " batch=" << batch_size
              << " sarray_ms=" << sarray_seconds * 1.e3
              << " view_batch_ms=" << batch_seconds * 1.e3
              << " hierarchical_tile1_ms=" << tile_one_seconds * 1.e3
              << " hierarchical_best_ms=" << best_hierarchical_seconds * 1.e3
              << " hierarchical_best_tile=" << best_hierarchical_tile
              << " tensorcore_warp1_ms=" << tensorcore_warp_one_seconds * 1.e3
              << " tensorcore_best_ms=" << best_tensorcore_seconds * 1.e3
              << " tensorcore_best_warps=" << best_tensorcore_warps
              << " tensorcore_max_abs_difference=" << tensorcore_error
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
