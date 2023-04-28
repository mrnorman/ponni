
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Gelu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
    };

    Params params;

    Gelu() {};
    Gelu(int num_inputs) {
      init(num_inputs);
    }


    char const * get_label         () const { return "Gelu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        real x = input(irow,ibatch);
        output(irow,ibatch) = x/2*(1+std::erf(x/std::sqrt(2)));
      }
    }


    void print_verbose() const { }


    int get_num_trainable_parameters() const { return 0; }


    doubleHost1d to_array() const {
      doubleHost1d data("Gelu_params",1);
      data(0) = params.num_inputs;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) );
    }


    void validate() const { }

  };

}


