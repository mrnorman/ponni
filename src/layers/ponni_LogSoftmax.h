#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct LogSoftmax {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = LogSoftmax<real,N,NewMemorySpace>;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, MemorySpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, MemorySpace> real2d;
    bool static constexpr overwrite_input = true;
    bool static constexpr binop = false;
    bool static constexpr save = false;

    // LogSoftmax depends on every feature and needs multiple passes, so it is a
    // fusion barrier. Custom cross-feature layers should make the same choice
    // unless Inference also implements a dedicated fused executor for them.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;
    int static constexpr INPUT_SIZE = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);
    struct Params { int num_inputs; };
    Params params;

    LogSoftmax() = default;
    ~LogSoftmax() = default;
    explicit LogSoftmax(int num_inputs) { init(num_inputs); }
    void init(int num_inputs) { params.num_inputs = num_inputs; }

    // Model creation may rebind a layer to another memory space. Layers
    // without Views only need to preserve their scalar configuration.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(params.num_inputs);
    }
    char const * get_label() const { return "LogSoftmax"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                           int ibatch, Params const & p) {
      ponni::require_layout_right_views<InputView,OutputView>();
      real max_value = input(0,ibatch);
      for (int i = 1; i < p.num_inputs; i++) {
        max_value = input(i,ibatch) > max_value ? input(i,ibatch) : max_value;
      }
      real sum_exp = static_cast<real>(0);
      for (int i = 0; i < p.num_inputs; i++) sum_exp += std::exp(input(i,ibatch) - max_value);
      real const log_sum_exp = std::log(sum_exp) + max_value;
      for (int i = 0; i < p.num_inputs; i++) output(i,ibatch) = input(i,ibatch) - log_sum_exp;
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & input,
                                                           ponni::SArray<real,N> & output, Params const &) {
      real max_value = input(0);
      for (int i = 1; i < N; i++) max_value = input(i) > max_value ? input(i) : max_value;
      real sum_exp = static_cast<real>(0);
      for (int i = 0; i < N; i++) sum_exp += std::exp(input(i) - max_value);
      real const log_sum_exp = std::log(sum_exp) + max_value;
      for (int i = 0; i < N; i++) output(i) = input(i) - log_sum_exp;
    }

    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }
    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: LogSoftmax num_inputs must be > 0");
    }
  };

}
