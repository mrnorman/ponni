
#pragma once
// Included by ponni.h

namespace ponni {

  class Prelu {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real alpha, threshold;
    };

    Params params;

    Prelu() {};
    Prelu(int num_inputs, real alpha=1.67326, real threshold=0) {
      init(num_inputs, alpha, threshold);
    }


    char const * get_label      () const { return "Prelu"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs, real alpha=1.67326, real threshold=0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.threshold   = threshold;
    }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      real alpha     = params.alpha;
      real threshold = params.threshold;
      real x         = input(irow,ibatch);
      if (x < threshold) { output(irow,ibatch) = alpha*x; }
      else               { output(irow,ibatch) = x;       }
    }


    void print_verbose() const {
      std::cout << "    alpha:     " << params.alpha     << "\n";
      std::cout << "    threshold: " << params.threshold << "\n";
    }

  };

}


