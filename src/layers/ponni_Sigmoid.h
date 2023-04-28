
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Sigmoid {
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;

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
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = static_cast<real>(1) / (1 + std::exp(-input(irow,ibatch)));
      }
    }


    void print_verbose() const { }


    void validate() const { }

  };

}


