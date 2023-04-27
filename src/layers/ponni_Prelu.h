
#pragma once
// Included by ponni.h

namespace ponni {

  class Prelu {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

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


    char const * get_label         () const { return "Prelu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    void init(int num_inputs, real alpha=1.67326, real threshold=0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.threshold   = threshold;
    }


    YAKL_INLINE void compute_all_outputs(real2d const &input, real2d const &output, int ibatch) const {
      for (int irow = 0; irow < params.num_outputs; irow++) {
        real alpha     = params.alpha;
        real threshold = params.threshold;
        real x         = input(irow,ibatch);
        if (x < threshold) { output(irow,ibatch) = alpha*x; }
        else               { output(irow,ibatch) = x;       }
      }
    }


    void print_verbose() const {
      std::cout << "    alpha:     " << params.alpha     << "\n";
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    void validate() const { }

  };

}


