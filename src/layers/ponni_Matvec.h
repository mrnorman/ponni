
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float,
            int N_IN = 1,
            int N_OUT = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Matvec {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Matvec<real,N_IN,N_OUT,NewMemorySpace>;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    // A dense layer starts a fused feed-forward block. Custom dense layers may
    // use this trait when they also provide the device-safe compute_output()
    // scalar interface below. The runtime feature dimensions remain in Params.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::dense;

    int static constexpr INPUT_SIZE  = static_cast<int>(N_IN );
    int static constexpr OUTPUT_SIZE = static_cast<int>(N_OUT);

    struct Params {
      real2d weights;
      bool   trainable;
    };

    Params params;

    Matvec () = default;
    ~Matvec() = default;
    template < class INIT = Initializer_Random_Uniform<real> >
    Matvec( int num_inputs , int num_outputs , bool trainable = true ,
            INIT initializer = Initializer_Random_Uniform<real>() ) {
      real2d weights("Bias_weights",num_inputs,num_outputs);
      initializer.fill( weights );
      init(weights,trainable);
    }
    Matvec( real2d const &weights , bool trainable=true ) { init(weights,trainable); }

    void init( real2d const &weights , bool trainable=true ) {
      if ( ! weights.is_allocated() ) Kokkos::abort("ERROR: Matvec weights matrix not is_allocated");
      params.weights   = weights;
      params.trainable = trainable;
    }

    // A memory-space rebound owns an independent copy of the weights. Custom
    // parameterized layers should follow this contract for every owned View.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & memory_space = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(
          ponni::create_memory_space_copy(params.weights, memory_space), params.trainable);
    }

    char const * get_label() const { return "Matvec"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.weights.extent(0); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.weights.extent(1); }
    int    get_num_inputs               () const { return params.weights.extent(0); }
    int    get_num_outputs              () const { return params.weights.extent(1); }
    int    get_num_trainable_parameters () const { return params.trainable ? params.weights.size() : 0; }

    // Compute one output without storing it. Inference uses this scalar result
    // to apply Bias and activation epilogues before touching workspace.
    template <class InputView>
    KOKKOS_INLINE_FUNCTION static real compute_output(InputView const & input, int irow, int ibatch,
                                                       Params const & params_in) {
      ponni::require_layout_right_views<InputView>();
      real value = 0;
      for (int k = 0; k < get_num_inputs(params_in); k++) {
        value += params_in.weights(k,irow) * input(k,ibatch);
      }
      return value;
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( InputView const & input     ,
                                                            OutputView const & output    ,
                                                            int            ibatch    ,
                                                            Params const & params_in ) {
      ponni::require_layout_right_views<InputView,OutputView>();
      int num_outputs = get_num_outputs(params_in);
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch) = compute_output(input, irow, ibatch, params_in);
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( ponni::SArray<real,N_IN > const & input     ,
                                                            ponni::SArray<real,N_OUT>       & output    ,
                                                            Params                    const & params_in ) {
      for (int irow = 0; irow < N_OUT; irow++) {
        real tmp = 0;
        for (int k=0; k < N_IN; k++) { tmp += params_in.weights(k,irow) * input(k); }
        output(irow) = tmp;
      }
    }

    void set_trainable_parameters(real1d const &in) {
      if (params.trainable) {
        if (in.extent(0) < get_num_trainable_parameters()) Kokkos::abort("ERROR: Matvec trainable input too small");
        auto in_reduced = Kokkos::subview(in,std::pair<int,int>(0,get_num_trainable_parameters()));
        Kokkos::deep_copy(ponni::flatten(params.weights),in_reduced);
      }
    }

    real1d get_trainable_parameters() const {
      if (params.trainable) return ponni::flatten(params.weights);
      return real1d();
    }

    void validate() const {
      if (! params.weights.is_allocated()) Kokkos::abort("ERROR: weights not is_allocated");
      if (params.weights.extent(0) == 0 || params.weights.extent(1) == 0) {
        Kokkos::abort("ERROR: Matvec weights dimensions must be nonzero");
      }
    }
  };

}
