

#pragma once
// Included by ponni.h

namespace ponni {

  class Silu {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
    };

    Params params;

    Silu() {};
    Silu(int num_inputs) {
      init(num_inputs);
    }


    char const * get_label      () const { return "Silu"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) / (1 + std::exp(-input(irow,ibatch)));
    }


    void print_verbose() const {
    }

  };

}

