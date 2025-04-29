
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N, class real = float>
  struct Binop_Concatenate {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = N;

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
    }


    char const * get_label() const { return "Binop_Concatenate"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_trainable_parameters () const { return 0; }
    int    get_array_representation_size() const { return 4; }


    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const &input1, real2d const &input2, real2d const &output,
                                                           int ibatch, Params const &params_in) {
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


    void set_trainable_parameters(real1d const &in) { }


    real1d get_trainable_parameters() const { return real1d(); }


    doubleHost1d to_array() const {
      doubleHost1d data("Binop_Concatenate_params",get_array_representation_size());
      data(0) = get_num_inputs ();
      data(1) = get_num_outputs();
      data(2) = params.after ? 1 : 0;
      data(3) = N;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      if (data(3) != N) Kokkos::abort("ERROR: Binop_Concatenate saved state index incompatible with data from file");
      init( static_cast<int>(data(0)) , static_cast<int>(data(1)) , data(2) == 1 );
    }


    void validate(int saved_layer_num_inputs) const {
      if ( params.num_outputs != saved_layer_num_inputs + params.num_inputs ) {
        Kokkos::abort("ERROR: Binop_Concatenate: this layer's num outputs != "
                         "this layer's num inputs + saved layer's num inputs");
      }
    }

  };

}


