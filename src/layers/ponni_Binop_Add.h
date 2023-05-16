
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N, class real = float>
  struct Binop_Add {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = N;

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
    YAKL_INLINE static int get_num_inputs   (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs  (Params const &params_in) { return params_in.num_outputs; }
    YAKL_INLINE static int get_num_ensembles(Params const &params_in) { return 1; }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_ensembles            () const { return 1; }
    int    get_num_trainable_parameters () const { return 0; }
    int    get_array_representation_size() const { return 2; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input1, real3d const &input2, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch,iens) = input1(irow,ibatch,iens) + input2(irow,ibatch,iens);
      }
    }


    void set_trainable_parameters(real2d const &in, bool fence = true) { if (fence) yakl::fence(); }


    real2d get_trainable_parameters() const { return real2d(); }


    doubleHost1d to_array() const {
      doubleHost1d data("Binary_Add_params",get_array_representation_size());
      data(0) = get_num_inputs();
      data(1) = N;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      if (data(1) != N) yakl::yakl_throw("ERROR: Binop_Add saved state index incompatible with data from file");
      init( static_cast<int>(data(0)) );
    }


    void validate(int saved_layer_num_inputs) const {
      if ( params.num_inputs != saved_layer_num_inputs ) {
        yakl::yakl_throw("ERROR: Binop_Add: Saved layer num inputs != this layer's num inputs");
      }
    }

  };

}


