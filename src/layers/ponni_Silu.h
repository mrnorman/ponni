
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Silu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
    };

    Params params;

    Silu() {};
    Silu( int num_inputs ) {
      init( num_inputs );
    }


    void init( int num_inputs ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    char const * get_label         () const { return "Silu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 1; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch,iens) = input(irow,ibatch,iens) / (1 + std::exp(-input(irow,ibatch,iens)));
      }
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Silu_params",1);
      data(0) = params.num_inputs;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) );
    }


    void validate() const { }

  };

}

