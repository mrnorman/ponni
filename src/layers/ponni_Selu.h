#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Selu {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Selu<real,N,NewMemorySpace>;
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> doubleHost1d;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, MemorySpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, MemorySpace> real2d;
    bool static constexpr overwrite_input = true;
    bool static constexpr binop = false;
    bool static constexpr save = false;
    int static constexpr INPUT_SIZE = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);
    struct Params { int num_inputs; };
    Params params;

    Selu() = default;
    ~Selu() = default;
    explicit Selu(int num_inputs) { init(num_inputs); }
    void init(int num_inputs) { params.num_inputs = num_inputs; }
    char const * get_label() const { return "SELU"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 1; }

    KOKKOS_INLINE_FUNCTION static real apply(real x) {
      real constexpr alpha = static_cast<real>(1.6732632423543772848);
      real constexpr scale = static_cast<real>(1.0507009873554804934);
      return x > static_cast<real>(0) ? scale * x : scale * alpha * (std::exp(x) - static_cast<real>(1));
    }
    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & in, OutputView const & out,
                                                           int ibatch, Params const & p) {
      for (int i = 0; i < p.num_inputs; i++) out(i,ibatch) = apply(in(i,ibatch));
    }
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & in,
                                                           ponni::SArray<real,N> & out, Params const &) {
      for (int i = 0; i < N; i++) out(i) = apply(in(i));
    }
    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }
    doubleHost1d to_array() const { doubleHost1d data("SELU_params", 1); data(0) = params.num_inputs; return data; }
    void from_array(doubleHost1d const & data) { init(static_cast<int>(data(0))); }
    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: SELU num_inputs must be > 0");
    }
  };

  template <class real = float,
            int N = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  using SELU = Selu<real,N,MemorySpace>;

}
