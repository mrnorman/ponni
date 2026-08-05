
#pragma once
// Included by ponni.h

namespace ponni {

  template <int ISAVE,
            class real = float,
            int N1 = 1,
            int N2 = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Binop_Concatenate {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Binop_Concatenate<ISAVE,real,N1,N2,NewMemorySpace>;
    typedef Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = ISAVE;

    int static constexpr INPUT_SIZE  = static_cast<int>(N1);
    int static constexpr OUTPUT_SIZE = static_cast<int>(N1+N2);

    struct Params {
      int  num_inputs;
      int  num_outputs;
      bool after;
    };

    Params params;

    Binop_Concatenate () = default;
    ~Binop_Concatenate() = default;
    Binop_Concatenate( int num_inputs , int num_outputs , bool after=true ) { init( num_inputs , num_outputs , after); }

    void init( int num_inputs , int num_outputs , bool after=true ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_outputs;
      params.after       = after;
    }

    char const * get_label() const { return "Binop_Concatenate"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_trainable_parameters () const { return 0; }
    int    get_array_representation_size() const { return 4; }

    template <class InputView1, class InputView2, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( InputView1 const & input1    ,
                                                            InputView2 const & input2    ,
                                                            OutputView const & output    ,
                                                            int            ibatch    ,
                                                            Params const & params_in ) {
      if (params_in.after) {
        int num_inputs_1 = input1.extent(0);
        int num_outputs = params_in.num_outputs;
        for (int irow = 0; irow < num_outputs; irow++) {
          output(irow,ibatch) = irow < num_inputs_1 ? input1(irow,ibatch) : input2(irow - num_inputs_1,ibatch);
        }
      } else {
        int num_inputs_2 = input2.extent(0);
        int num_outputs = params_in.num_outputs;
        for (int irow = 0; irow < num_outputs; irow++) {
          output(irow,ibatch) = irow < num_inputs_2 ? input2(irow,ibatch) : input1(irow - num_inputs_2,ibatch);
        }
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( ponni::SArray<real,N1   > const & input1    ,
                                                            ponni::SArray<real,N2   > const & input2    ,
                                                            ponni::SArray<real,N1+N2>       & output    ,
                                                            Params                    const & params_in ) {
      if (params_in.after) {
        for (int i = 0; i < N1+N2; i++) { output(i) = i < N1 ? input1(i) : input2(i - N1); }
      } else {
        for (int i = 0; i < N1+N2; i++) { output(i) = i < N2 ? input2(i) : input1(i - N2); }
      }
    }

    void set_trainable_parameters(real1d const &in) { }

    real1d get_trainable_parameters() const { return real1d(); }

    doubleHost1d to_array() const {
      doubleHost1d data("Binop_Concatenate_params",get_array_representation_size());
      data(0) = get_num_inputs ();
      data(1) = get_num_outputs();
      data(2) = params.after ? 1 : 0;
      data(3) = ISAVE;
      return data;
    }

    void from_array(doubleHost1d const &data) {
      if (data(3) != ISAVE) Kokkos::abort("ERROR: Binop_Concatenate saved state index incompatible with data from file");
      init( static_cast<int>(data(0)) , static_cast<int>(data(1)) , data(2) == 1 );
    }

    void validate(int saved_layer_num_inputs) const {
      if (params.num_inputs <= 0 || saved_layer_num_inputs <= 0) {
        Kokkos::abort("ERROR: Binop_Concatenate input sizes must be > 0");
      }
      if ( params.num_outputs != saved_layer_num_inputs + params.num_inputs ) {
        Kokkos::abort("ERROR: Binop_Concatenate: this layer's num outputs != "
                         "this layer's num inputs + saved layer's num inputs");
      }
    }
  };

}
