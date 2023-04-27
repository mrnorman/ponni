
#pragma once
// Included by ponni.h

namespace ponni {

  class Gelu {
  public:

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


    YAKL_INLINE void compute_all_outputs(real2d const &input, real2d const &output, int ibatch) const {
      for (int irow = 0; irow < params.num_outputs; irow++) {
        real x = input(irow,ibatch);
        output(irow,ibatch) = x/2*(1+std::erf(x/std::sqrt(2)));
      }
    }


    void print_verbose() const { }


    void validate() const { }

  };

}


