#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float, int N = 1>
  struct MinMaxNorm {
    typedef Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,ponni::DeviceSpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,ponni::DeviceSpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false;
    bool static constexpr save            = false;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params {
      int  num_inputs;
      real out_min;
      real out_max;
      real epsilon;
    };

    Params params;

    MinMaxNorm() = default;
    ~MinMaxNorm() = default;
    MinMaxNorm(int num_inputs,
               real out_min = static_cast<real>(0),
               real out_max = static_cast<real>(1),
               real epsilon = static_cast<real>(1.e-12)) {
      init(num_inputs, out_min, out_max, epsilon);
    }

    void init(int num_inputs,
              real out_min = static_cast<real>(0),
              real out_max = static_cast<real>(1),
              real epsilon = static_cast<real>(1.e-12)) {
      params.num_inputs = num_inputs;
      params.out_min = out_min;
      params.out_max = out_max;
      params.epsilon = epsilon;
    }

    char const * get_label() const { return "MinMaxNorm"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 4; }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const & input,
                                                           real2d const & output,
                                                           int ibatch,
                                                           Params const & params_in) {
      int n = params_in.num_inputs;
      real min_v = input(0,ibatch);
      real max_v = input(0,ibatch);
      for (int i = 1; i < n; i++) {
        if (input(i,ibatch) < min_v) min_v = input(i,ibatch);
        if (input(i,ibatch) > max_v) max_v = input(i,ibatch);
      }
      real scale = (params_in.out_max - params_in.out_min) / ((max_v - min_v) + params_in.epsilon);
      for (int i = 0; i < n; i++) {
        output(i,ibatch) = params_in.out_min + (input(i,ibatch) - min_v) * scale;
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & input,
                                                           ponni::SArray<real,N> & output,
                                                           Params const & params_in) {
      int n = params_in.num_inputs;
      real min_v = input(0);
      real max_v = input(0);
      for (int i = 1; i < n; i++) {
        if (input(i) < min_v) min_v = input(i);
        if (input(i) > max_v) max_v = input(i);
      }
      real scale = (params_in.out_max - params_in.out_min) / ((max_v - min_v) + params_in.epsilon);
      for (int i = 0; i < n; i++) {
        output(i) = params_in.out_min + (input(i) - min_v) * scale;
      }
    }

    void set_trainable_parameters(real1d const & in) { }
    real1d get_trainable_parameters() const { return real1d(); }

    doubleHost1d to_array() const {
      doubleHost1d data("MinMaxNorm_array", get_array_representation_size());
      data(0) = params.num_inputs;
      data(1) = params.out_min;
      data(2) = params.out_max;
      data(3) = params.epsilon;
      return data;
    }

    void from_array(doubleHost1d const & data) {
      init(static_cast<int>(data(0)), static_cast<real>(data(1)), static_cast<real>(data(2)), static_cast<real>(data(3)));
    }

    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: MinMaxNorm num_inputs must be > 0");
      if (params.out_max <= params.out_min) Kokkos::abort("ERROR: MinMaxNorm requires out_max > out_min");
      if (params.epsilon <= static_cast<real>(0)) Kokkos::abort("ERROR: MinMaxNorm epsilon must be > 0");
    }
  };

}
