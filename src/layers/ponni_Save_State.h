
#pragma once
// Included by ponni.h

namespace ponni {

  template <int ISAVE, class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Save_State {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Save_State<ISAVE,real,N,NewMemorySpace>;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = true;

    // Saving is an observable materialization point used by a later branch, so
    // it must remain a fusion barrier. Custom layers with side effects or
    // externally consumed state must also remain barriers.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;
    int  static constexpr index           = ISAVE;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Save_State () = default;
    ~Save_State() = default;
    Save_State( int num_inputs ) { init( num_inputs ); }

    void init( int num_inputs ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }

    // Model creation may rebind a layer to another memory space. Layers
    // without Views only need to preserve their scalar configuration.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(params.num_inputs);
    }

    char const * get_label() const { return "Save_State"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_trainable_parameters () const { return 0; }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( InputView const & input     ,
                                                            OutputView const & output    ,
                                                            int            ibatch    ,
                                                            Params const & params_in ) {
      ponni::require_layout_right_views<InputView,OutputView>();
      int num_outputs = params_in.num_outputs;
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch);
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( ponni::SArray<real,N> const & input     ,
                                                            ponni::SArray<real,N>       & output    ,
                                                            Params                const & params_in ) {
      for (int i = 0; i < N; i++) { output(i) = input(i); }
    }

    void set_trainable_parameters(real1d const &in) { }

    real1d get_trainable_parameters() const { return real1d(); }

    void validate() const {
      if (params.num_inputs <= 0) Kokkos::abort("ERROR: Save_State num_inputs must be > 0");
      if (params.num_outputs != params.num_inputs) Kokkos::abort("ERROR: Save_State input/output size mismatch");
    }
  };

}
