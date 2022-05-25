
#pragma once
// Included by ponni.h

namespace ponni {

  class Sigmoid {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
    };

    Params params;

    Sigmoid() {};
    Sigmoid(int num_inputs) {
      init(num_inputs);
    }


    char const * get_label         () const { return "Sigmoid"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    YAKL_INLINE void compute_all_outputs(real2d const &input, real2d const &output, int ibatch) const {
      for (int irow = 0; irow < params.num_outputs; irow++) { compute_one_output(input, output, ibatch, irow); }
    }


    YAKL_INLINE void compute_one_output(real2d const &input, real2d const &output, int ibatch, int irow) const {
      output(irow,ibatch) = static_cast<real>(1) / (1 + std::exp(-input(irow,ibatch)));
    }


    void print_verbose() const { }


    void validate() const { }

  };

}


