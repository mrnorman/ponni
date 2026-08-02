#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1>
  struct Softplus {
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> doubleHost1d;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, ponni::DeviceSpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, ponni::DeviceSpace> real2d;
    bool static constexpr overwrite_input = true;
    bool static constexpr binop = false;
    bool static constexpr save = false;
    int static constexpr INPUT_SIZE = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);
    struct Params { int num_inputs; real beta; };
    Params params;

    Softplus() = default;
    ~Softplus() = default;
    explicit Softplus(int num_inputs, real beta = static_cast<real>(1)) { init(num_inputs, beta); }
    void init(int num_inputs, real beta = static_cast<real>(1)) { params = {num_inputs, beta}; }
    char const * get_label() const { return "Softplus"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 2; }
    KOKKOS_INLINE_FUNCTION static real apply(real x, Params const & p) {
      real const scaled_x = p.beta * x;
      if (scaled_x > static_cast<real>(0)) {
        return (scaled_x + std::log(static_cast<real>(1) + std::exp(-scaled_x))) / p.beta;
      }
      return std::log(static_cast<real>(1) + std::exp(scaled_x)) / p.beta;
    }
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const & in, real2d const & out,
                                                           int ibatch, Params const & p) {
      for (int i = 0; i < p.num_inputs; i++) out(i,ibatch) = apply(in(i,ibatch), p);
    }
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & in,
                                                           ponni::SArray<real,N> & out, Params const & p) {
      for (int i = 0; i < N; i++) out(i) = apply(in(i), p);
    }
    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }
    doubleHost1d to_array() const {
      doubleHost1d data("Softplus_params", 2); data(0) = params.num_inputs; data(1) = params.beta; return data;
    }
    void from_array(doubleHost1d const & data) { init(static_cast<int>(data(0)), static_cast<real>(data(1))); }
    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: Softplus num_inputs must be > 0");
      if (params.beta <= static_cast<real>(0)) Kokkos::abort("ERROR: Softplus beta must be > 0");
    }
  };

}
