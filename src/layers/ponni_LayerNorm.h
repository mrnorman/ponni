#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1>
  struct LayerNorm {
    typedef Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,Kokkos::HostSpace > realHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,ponni::DeviceSpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,ponni::DeviceSpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false;
    bool static constexpr save            = false;

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

    char const * get_label() const { return "LayerNorm"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.gamma.extent(0); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.gamma.extent(0); }
    int get_num_inputs() const { return params.gamma.extent(0); }
    int get_num_outputs() const { return params.gamma.extent(0); }
    int get_num_trainable_parameters() const { return params.trainable ? 2 * params.gamma.size() : 0; }
    int get_array_representation_size() const { return 3 + 2 * params.gamma.size(); }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const & input,
                                                           real2d const & output,
                                                           int ibatch,
                                                           Params const & params_in) {
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

    doubleHost1d to_array() const {
      doubleHost1d data("LayerNorm_array", get_array_representation_size());
      int n = params.gamma.extent(0);
      data(0) = n;
      data(1) = params.trainable ? 1 : 0;
      data(2) = params.epsilon;
      auto gamma_h = ponni::create_host_copy(params.gamma);
      auto beta_h = ponni::create_host_copy(params.beta);
      for (int i = 0; i < n; i++) data(3 + i) = gamma_h(i);
      for (int i = 0; i < n; i++) data(3 + n + i) = beta_h(i);
      return data;
    }

    void from_array(doubleHost1d const & data) {
      int n = static_cast<int>(data(0));
      bool trainable = data(1) == 1;
      real eps = static_cast<real>(data(2));
      realHost1d gamma_h("LayerNorm_gamma_h", n);
      realHost1d beta_h("LayerNorm_beta_h", n);
      for (int i = 0; i < n; i++) gamma_h(i) = static_cast<real>(data(3 + i));
      for (int i = 0; i < n; i++) beta_h(i) = static_cast<real>(data(3 + n + i));
      init(ponni::create_device_copy(gamma_h), ponni::create_device_copy(beta_h), eps, trainable);
    }

    void validate() const {
      if (!params.gamma.is_allocated() || !params.beta.is_allocated()) Kokkos::abort("ERROR: LayerNorm params not allocated");
      if (params.gamma.extent(0) == 0) Kokkos::abort("ERROR: LayerNorm params must not be empty");
      if (params.gamma.extent(0) != params.beta.extent(0)) Kokkos::abort("ERROR: LayerNorm gamma/beta size mismatch");
      if (params.epsilon <= static_cast<real>(0)) Kokkos::abort("ERROR: LayerNorm epsilon must be > 0");
    }
  };

}
