
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N, class real = float>
  struct Binop_Add {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = N;

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Binop_Add() {}
    Binop_Add( int num_inputs ) {
      init( num_inputs );
    }


    void init( int num_inputs ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    char const * get_label         () const { return "Binop_Add"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input1, real2d const &input2, real2d const &output,
                                         int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = input1(irow,ibatch) + input2(irow,ibatch);
      }
    }


    void print_verbose() const {
      std::cout << "    adding from saved index: " << index << "\n";
    }


    int get_num_trainable_parameters() const { return 0; }


    doubleHost1d to_array() const {
      doubleHost1d data("Binary_Add_params",1);
      data(0) = params.num_inputs;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) );
    }


    void validate(int saved_layer_num_inputs) const {
      if ( params.num_inputs != saved_layer_num_inputs ) {
        yakl::yakl_throw("ERROR: Binop_Add: Saved layer num inputs != this layer's num inputs");
      }
    }

  };

}


