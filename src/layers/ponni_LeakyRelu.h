#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct LeakyRelu {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = LeakyRelu<real,N,NewMemorySpace>;
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> doubleHost1d;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, MemorySpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, MemorySpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false;
    bool static constexpr save            = false;

    // Pointwise permits apply() to run on a dense accumulator before it is
    // written. Custom layers must preserve feature count and have no
    // cross-feature dependency to make the same declaration safely.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::pointwise;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params { int num_inputs; real negative_slope; };
    Params params;

    LeakyRelu() = default;
    ~LeakyRelu() = default;
    explicit LeakyRelu(int num_inputs, real negative_slope = static_cast<real>(0.01)) {
      init(num_inputs, negative_slope);
    }

    void init(int num_inputs, real negative_slope = static_cast<real>(0.01)) {
      params.num_inputs = num_inputs;
      params.negative_slope = negative_slope;
    }

    char const * get_label() const { return "LeakyReLU"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 2; }

    KOKKOS_INLINE_FUNCTION static real apply(real x, Params const & params_in) {
      return x > static_cast<real>(0) ? x : params_in.negative_slope * x;
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                           int ibatch, Params const & params_in) {
      for (int i = 0; i < params_in.num_inputs; i++) output(i,ibatch) = apply(input(i,ibatch), params_in);
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & input,
                                                           ponni::SArray<real,N> & output,
                                                           Params const & params_in) {
      for (int i = 0; i < N; i++) output(i) = apply(input(i), params_in);
    }

    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }

    doubleHost1d to_array() const {
      doubleHost1d data("LeakyReLU_params", get_array_representation_size());
      data(0) = params.num_inputs;
      data(1) = params.negative_slope;
      return data;
    }

    void from_array(doubleHost1d const & data) {
      init(static_cast<int>(data(0)), static_cast<real>(data(1)));
    }

    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: LeakyReLU num_inputs must be > 0");
    }
  };

  template <class real = float,
            int N = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  using LeakyReLU = LeakyRelu<real,N,MemorySpace>;

}
