
#pragma once
// Included by ponni.h

namespace ponni {

  class Binary_Step {
  public:

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


    char const * get_label      () const { return "Binary Step"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs, real threshold = 0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.threshold   = threshold ;
    }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) >= params.threshold ? 1 : 0;
    }


    void print_verbose() const {
      std::cout << "    threshold: " << params.threshold << "\n";
    }

  };

}


