
#pragma once
// Included by ponni.h

namespace ponni {

  template <int ISAVE, class real = float, int N = 1, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Binop_Add {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Binop_Add<ISAVE,real,N,NewMemorySpace>;
    typedef Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;

    // This layer consumes a separately saved state, which makes it a fusion
    // barrier in the general tuple traversal. Custom multi-input layers should
    // remain barriers until their saved-state lifetime is explicitly planned.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;
    int  static constexpr index           = ISAVE;

    int static constexpr INPUT_SIZE  = static_cast<int>(N);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N);

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Binop_Add () = default;
    ~Binop_Add() = default;
    Binop_Add( int num_inputs ) { init( num_inputs ); }

    void init( int num_inputs ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }

    char const * get_label() const { return "Binop_Add"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_trainable_parameters () const { return 0; }
    int    get_array_representation_size() const { return 2; }

    template <class InputView1, class InputView2, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( InputView1 const & input1    ,
                                                            InputView2 const & input2    ,
                                                            OutputView const & output    ,
                                                            int            ibatch    ,
                                                            Params const & params_in ) {
      int num_outputs = params_in.num_outputs;
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch) = input1(irow,ibatch) + input2(irow,ibatch);
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( ponni::SArray<real,N> const & input1    ,
                                                            ponni::SArray<real,N> const & input2    ,
                                                            ponni::SArray<real,N>       & output    ,
                                                            Params                const & params_in ) {
      for (int i = 0; i < N; i++) { output(i) = input1(i) + input2(i); }
    }

    void set_trainable_parameters(real1d const &in) { }

    real1d get_trainable_parameters() const { return real1d(); }

    doubleHost1d to_array() const {
      doubleHost1d data("Binary_Add_params",get_array_representation_size());
      data(0) = get_num_inputs();
      data(1) = ISAVE;
      return data;
    }

    void from_array(doubleHost1d const &data) {
      if (data(1) != ISAVE) Kokkos::abort("ERROR: Binop_Add saved state index incompatible with data from file");
      init( static_cast<int>(data(0)) );
    }

    void validate(int saved_layer_num_inputs) const {
      if (params.num_inputs <= 0 || params.num_outputs != params.num_inputs) {
        Kokkos::abort("ERROR: Binop_Add invalid input/output size");
      }
      if ( params.num_inputs != saved_layer_num_inputs ) {
        Kokkos::abort("ERROR: Binop_Add: Saved layer num inputs != this layer's num inputs");
      }
    }
  };

}
