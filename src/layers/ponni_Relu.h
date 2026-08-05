#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Relu {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Relu<real,N,NewMemorySpace>;
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> doubleHost1d;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, MemorySpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, MemorySpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false;
    bool static constexpr save            = false;

    // This activation is independently applicable to each scalar, so the
    // dynamic inference path may fold apply() into a preceding dense output.
    // Custom layers should use pointwise only when they preserve feature count
    // and never inspect another feature; otherwise use the default barrier.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::pointwise;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params { int num_inputs; };
    Params params;

    Relu() = default;
    ~Relu() = default;
    explicit Relu(int num_inputs) { init(num_inputs); }

    void init(int num_inputs) { params.num_inputs = num_inputs; }

    char const * get_label() const { return "ReLU"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 1; }

    KOKKOS_INLINE_FUNCTION static real apply(real x) {
      return x > static_cast<real>(0) ? x : static_cast<real>(0);
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                           int ibatch, Params const & params_in) {
      for (int i = 0; i < params_in.num_inputs; i++) output(i,ibatch) = apply(input(i,ibatch));
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & input,
                                                           ponni::SArray<real,N> & output, Params const &) {
      for (int i = 0; i < N; i++) output(i) = apply(input(i));
    }

    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }

    doubleHost1d to_array() const {
      doubleHost1d data("ReLU_params", get_array_representation_size());
      data(0) = params.num_inputs;
      return data;
    }

    void from_array(doubleHost1d const & data) { init(static_cast<int>(data(0))); }
    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: ReLU num_inputs must be > 0");
    }
  };

  template <class real = float,
            int N = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  using ReLU = Relu<real,N,MemorySpace>;

}
