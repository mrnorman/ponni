
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Binary_Step {
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real threshold;
    };

    Params params;

    Binary_Step() {};
    Binary_Step(int num_inputs , real threshold = 0) {
      init(num_inputs, threshold);
    }


    char const * get_label         () const { return "Binary Step"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    void init(int num_inputs, real threshold = 0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.threshold   = threshold ;
    }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch) >= params_in.threshold ? 1 : 0;
      }
    }

    void print_verbose() const {
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    void validate() const { }

  };

}


