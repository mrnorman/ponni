#include "Branch4W32D2.hpp"
#include "Branch4W64D2.hpp"
#include "LongSkipW32D8.hpp"
#include "LongSkipW64D8.hpp"
#include "ResidualW128D8.hpp"
#include "ResidualW32D4.hpp"
#include "ResidualW32D8.hpp"
#include "ResidualW64D4.hpp"
#include "ResidualW64D8.hpp"
#include "SeqW128D2.hpp"
#include "SeqW16D2.hpp"
#include "SeqW32D2.hpp"
#include "SeqW32D8.hpp"
#include "SeqW64D2.hpp"
#include "SeqW64D8.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int constexpr batch_size = 1000000;
int constexpr timed_runs = 5;

template <class Function>
double time_inference(Function const & function) {
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int run = 0; run < timed_runs; run++) function();
  Kokkos::fence();
  return timer.seconds() / timed_runs;
}

template <class Model>
bool run_batch_team(Model const & model, typename Model::InputView const & inputs,
                    typename Model::OutputView const & outputs, int team_size) {
  switch (team_size) {
    case 64: return model.try_infer_batch_team_64(inputs,outputs);
    case 128: return model.try_infer_batch_team_128(inputs,outputs);
    case 256: return model.try_infer_batch_team_256(inputs,outputs);
    case 512: return model.try_infer_batch_team_512(inputs,outputs);
    case 1024: return model.try_infer_batch_team_1024(inputs,outputs);
    default: return false;
  }
}

template <class Model>
bool benchmark(std::string const & root, char const * name, char const * architecture, int width, int depth) {
  Model model;
  std::string error;
  if (!model.load_weights(root + "/generated/" + name + "/weights.bin", &error)) {
    std::cerr << name << ": " << error << std::endl;
    return false;
  }

  typename Model::InputView inputs("architecture_inputs", Model::num_inputs, batch_size);
  typename Model::OutputView batch_outputs("architecture_batch_outputs", Model::num_outputs, batch_size);
  typename Model::OutputView batch_team_outputs("architecture_batch_team_outputs", Model::num_outputs, batch_size);
  Kokkos::parallel_for(
      "architecture_initialize_inputs",
      Kokkos::RangePolicy<typename Model::execution_space>(0, Model::num_inputs * batch_size),
      KOKKOS_LAMBDA(int linear) {
        int const ibatch = linear % batch_size;
        int const iinput = linear / batch_size;
        inputs(iinput,ibatch) = 0.01f * ((iinput + ibatch) % 31 - 15);
      });

  model.infer_batch(inputs,batch_outputs);
  Kokkos::fence();
  double const batch_seconds = time_inference([&]() { model.infer_batch(inputs,batch_outputs); });
  bool passed = true;
  for (int team_size : {64, 128, 256, 512, 1024}) {
    if (!run_batch_team(model,inputs,batch_team_outputs,team_size)) {
      std::cout << "architecture_result name=" << name
                << " architecture=" << architecture
                << " width=" << width
                << " depth=" << depth
                << " batch_ms=" << batch_seconds * 1.e3
                << " team_size=" << team_size
                << " supported=0" << std::endl;
      continue;
    }
    Kokkos::fence();
    double const batch_team_seconds = time_inference(
        [&]() { (void) run_batch_team(model,inputs,batch_team_outputs,team_size); });
    float maximum_error = 0;
    std::size_t const output_elements = static_cast<std::size_t>(Model::num_outputs) * batch_size;
    Kokkos::parallel_reduce(
        "architecture_batch_team_error",
        Kokkos::RangePolicy<typename Model::execution_space>(0, output_elements),
        KOKKOS_LAMBDA(std::size_t linear, float & error_max) {
          int const ibatch = static_cast<int>(linear % batch_size);
          int const ioutput = static_cast<int>(linear / batch_size);
          float const value = Kokkos::abs(batch_outputs(ioutput,ibatch) - batch_team_outputs(ioutput,ibatch));
          if (value > error_max) error_max = value;
        },
        Kokkos::Max<float>(maximum_error));
    std::cout << "architecture_result name=" << name
              << " architecture=" << architecture
              << " width=" << width
              << " depth=" << depth
              << " batch_ms=" << batch_seconds * 1.e3
              << " team_size=" << team_size
              << " supported=1"
              << " batch_team_ms=" << batch_team_seconds * 1.e3
              << " speedup=" << batch_seconds / batch_team_seconds
              << " max_abs_difference=" << maximum_error << std::endl;
    if (maximum_error > 2.e-5f || !std::isfinite(maximum_error)) passed = false;
  }
  return passed;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " experiment_root" << std::endl;
    return 2;
  }
  Kokkos::initialize(argc,argv);
  ponni::init_device_pool(2ULL * 1024ULL * 1024ULL * 1024ULL);
  bool passed = true;
  {
    using namespace ponni::generated;
    std::string const root(argv[1]);
    passed = benchmark<SeqW16D2<float>>(root, "SeqW16D2", "sequential", 16, 2) && passed;
    passed = benchmark<SeqW32D2<float>>(root, "SeqW32D2", "sequential", 32, 2) && passed;
    passed = benchmark<SeqW64D2<float>>(root, "SeqW64D2", "sequential", 64, 2) && passed;
    passed = benchmark<SeqW128D2<float>>(root, "SeqW128D2", "sequential", 128, 2) && passed;
    passed = benchmark<SeqW32D8<float>>(root, "SeqW32D8", "sequential", 32, 8) && passed;
    passed = benchmark<SeqW64D8<float>>(root, "SeqW64D8", "sequential", 64, 8) && passed;
    passed = benchmark<ResidualW32D4<float>>(root, "ResidualW32D4", "residual", 32, 4) && passed;
    passed = benchmark<ResidualW64D4<float>>(root, "ResidualW64D4", "residual", 64, 4) && passed;
    passed = benchmark<ResidualW32D8<float>>(root, "ResidualW32D8", "residual", 32, 8) && passed;
    passed = benchmark<ResidualW64D8<float>>(root, "ResidualW64D8", "residual", 64, 8) && passed;
    passed = benchmark<ResidualW128D8<float>>(root, "ResidualW128D8", "residual", 128, 8) && passed;
    passed = benchmark<LongSkipW32D8<float>>(root, "LongSkipW32D8", "long_skip", 32, 8) && passed;
    passed = benchmark<LongSkipW64D8<float>>(root, "LongSkipW64D8", "long_skip", 64, 8) && passed;
    passed = benchmark<Branch4W32D2<float>>(root, "Branch4W32D2", "branch4", 32, 2) && passed;
    passed = benchmark<Branch4W64D2<float>>(root, "Branch4W64D2", "branch4", 64, 2) && passed;
  }
  ponni::finalize_device_pool();
  Kokkos::finalize();
  return passed ? 0 : 1;
}
