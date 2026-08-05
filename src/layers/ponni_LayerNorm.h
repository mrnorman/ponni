#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct LayerNorm {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = LayerNorm<real,N,NewMemorySpace>;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false;
    bool static constexpr save            = false;

    // LayerNorm needs statistics from the complete input feature vector, so it
    // is a fusion barrier. A custom reduction or normalization should remain a
    // barrier until it has an explicit whole-vector fusion implementation.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params {
      real1d gamma;
      real1d beta;
      real   epsilon;
      bool   trainable;
    };

    Params params;

    LayerNorm() = default;
    ~LayerNorm() = default;

    LayerNorm(int num_inputs, real epsilon = static_cast<real>(1.e-5), bool trainable = true) {
      real1d gamma("LayerNorm_gamma", num_inputs);
      real1d beta("LayerNorm_beta", num_inputs);
      Kokkos::deep_copy(gamma, static_cast<real>(1));
      Kokkos::deep_copy(beta, static_cast<real>(0));
      init(gamma, beta, epsilon, trainable);
    }

    LayerNorm(real1d const & gamma, real1d const & beta, real epsilon = static_cast<real>(1.e-5), bool trainable = true) {
      init(gamma, beta, epsilon, trainable);
    }

    void init(real1d const & gamma, real1d const & beta, real epsilon = static_cast<real>(1.e-5), bool trainable = true) {
      if (!gamma.is_allocated() || !beta.is_allocated()) Kokkos::abort("ERROR: LayerNorm gamma/beta not allocated");
      if (gamma.extent(0) != beta.extent(0)) Kokkos::abort("ERROR: LayerNorm gamma and beta size mismatch");
      params.gamma = gamma;
      params.beta = beta;
      params.epsilon = epsilon;
      params.trainable = trainable;
    }

    // Rebinding copies all owned parameter Views and preserves scalar
    // configuration. This is the extension contract for custom layers too.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & memory_space = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(
          ponni::create_memory_space_copy(params.gamma, memory_space),
          ponni::create_memory_space_copy(params.beta, memory_space), params.epsilon, params.trainable);
    }

    char const * get_label() const { return "LayerNorm"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.gamma.extent(0); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.gamma.extent(0); }
    int get_num_inputs() const { return params.gamma.extent(0); }
    int get_num_outputs() const { return params.gamma.extent(0); }
    int get_num_trainable_parameters() const { return params.trainable ? 2 * params.gamma.size() : 0; }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input,
                                                           OutputView const & output,
                                                           int ibatch,
                                                           Params const & params_in) {
      ponni::require_layout_right_views<InputView,OutputView>();
      int n = get_num_outputs(params_in);
      real mean = static_cast<real>(0);
      for (int i = 0; i < n; i++) mean += input(i,ibatch);
      mean /= static_cast<real>(n);

      real var = static_cast<real>(0);
      for (int i = 0; i < n; i++) {
        real d = input(i,ibatch) - mean;
        var += d * d;
      }
      var /= static_cast<real>(n);
      real inv_std = static_cast<real>(1) / std::sqrt(var + params_in.epsilon);

      for (int i = 0; i < n; i++) {
        real xhat = (input(i,ibatch) - mean) * inv_std;
        output(i,ibatch) = params_in.gamma(i) * xhat + params_in.beta(i);
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & input,
                                                           ponni::SArray<real,N> & output,
                                                           Params const & params_in) {
      int n = params_in.gamma.extent(0);
      real mean = static_cast<real>(0);
      for (int i = 0; i < n; i++) mean += input(i);
      mean /= static_cast<real>(n);

      real var = static_cast<real>(0);
      for (int i = 0; i < n; i++) {
        real d = input(i) - mean;
        var += d * d;
      }
      var /= static_cast<real>(n);
      real inv_std = static_cast<real>(1) / std::sqrt(var + params_in.epsilon);

      for (int i = 0; i < n; i++) {
        real xhat = (input(i) - mean) * inv_std;
        output(i) = params_in.gamma(i) * xhat + params_in.beta(i);
      }
    }

    void set_trainable_parameters(real1d const & in) {
      if (params.trainable) {
        int n = params.gamma.extent(0);
        if (in.extent(0) < 2 * n) Kokkos::abort("ERROR: LayerNorm trainable input too small");
        Kokkos::deep_copy(params.gamma, Kokkos::subview(in, std::pair<int,int>(0, n)));
        Kokkos::deep_copy(params.beta , Kokkos::subview(in, std::pair<int,int>(n, 2 * n)));
      }
    }

    real1d get_trainable_parameters() const {
      if (!params.trainable) return real1d();
      int n = params.gamma.extent(0);
      real1d out("LayerNorm_trainable", 2 * n);
      Kokkos::deep_copy(Kokkos::subview(out, std::pair<int,int>(0, n)), params.gamma);
      Kokkos::deep_copy(Kokkos::subview(out, std::pair<int,int>(n, 2 * n)), params.beta);
      return out;
    }

    void validate() const {
      if (!params.gamma.is_allocated() || !params.beta.is_allocated()) Kokkos::abort("ERROR: LayerNorm params not allocated");
      if (params.gamma.extent(0) == 0) Kokkos::abort("ERROR: LayerNorm params must not be empty");
      if (params.gamma.extent(0) != params.beta.extent(0)) Kokkos::abort("ERROR: LayerNorm gamma/beta size mismatch");
      if (params.epsilon <= static_cast<real>(0)) Kokkos::abort("ERROR: LayerNorm epsilon must be > 0");
    }
  };

}
