#pragma once
// Included by ponni.h

#include <cmath>

namespace ponni {

  template <class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Gelu {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Gelu<real,N,NewMemorySpace>;
    typedef Kokkos::View<real *, Kokkos::LayoutRight, MemorySpace> real1d;
    typedef Kokkos::View<real **, Kokkos::LayoutRight, MemorySpace> real2d;
    bool static constexpr overwrite_input = true;
    bool static constexpr binop = false;
    bool static constexpr save = false;

    // Pointwise permits apply() to run on a dense accumulator before it is
    // written. Custom layers must preserve feature count and have no
    // cross-feature dependency to make the same declaration safely.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::pointwise;
    int static constexpr INPUT_SIZE = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params { int num_inputs; bool approximate; };
    Params params;

    Gelu() = default;
    ~Gelu() = default;
    explicit Gelu(int num_inputs, bool approximate = false) { init(num_inputs, approximate); }
    void init(int num_inputs, bool approximate = false) { params = {num_inputs, approximate}; }

    // Model creation may rebind a layer to another memory space. Layers
    // without Views only need to preserve their scalar configuration.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(params.num_inputs, params.approximate);
    }

    char const * get_label() const { return "GELU"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
    int get_num_inputs() const { return params.num_inputs; }
    int get_num_outputs() const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }

    KOKKOS_INLINE_FUNCTION static real apply(real x, Params const & p) {
      if (p.approximate) {
        real constexpr sqrt_2_over_pi = static_cast<real>(0.79788456080286535588);
        real const x3 = x * x * x;
        return static_cast<real>(0.5) * x *
               (static_cast<real>(1) +
                std::tanh(sqrt_2_over_pi * (x + static_cast<real>(0.044715) * x3)));
      }
      real constexpr sqrt_2 = static_cast<real>(1.4142135623730950488);
      return static_cast<real>(0.5) * x * (static_cast<real>(1) + std::erf(x / sqrt_2));
    }
    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & in, OutputView const & out,
                                                           int ibatch, Params const & p) {
      ponni::require_layout_right_views<InputView,OutputView>();
      for (int i = 0; i < p.num_inputs; i++) out(i,ibatch) = apply(in(i,ibatch), p);
    }
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,N> const & in,
                                                           ponni::SArray<real,N> & out, Params const & p) {
      for (int i = 0; i < N; i++) out(i) = apply(in(i), p);
    }
    void set_trainable_parameters(real1d const &) { }
    real1d get_trainable_parameters() const { return real1d(); }
    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: GELU num_inputs must be > 0");
    }
  };

  template <class real = float,
            int N = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  using GELU = Gelu<real,N,MemorySpace>;

}
