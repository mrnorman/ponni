
#pragma once
// Included by ponni.h

namespace ponni {

  class Selu {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real alpha, lambda, threshold;
    };

    Params params;

    Selu() {};
    Selu(int num_inputs, real alpha=1.67326, real lambda=1.0507, real threshold=0) {
      init(num_inputs, alpha, lambda, threshold);
    }


    char const * get_label         () const { return "Selu"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs, real alpha=1.67326, real lambda=1.0507, real threshold=0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.lambda      = lambda;
      params.threshold   = threshold;
    }


    YAKL_INLINE void compute_all_outputs(real2d const &input, real2d const &output, int ibatch) const {
      for (int irow = 0; irow < params.num_outputs; irow++) { compute_one_output(input, output, ibatch, irow); }
    }


    YAKL_INLINE void compute_one_output(real2d const &input, real2d const &output, int ibatch, int irow) const {
      real alpha     = params.alpha;
      real lambda    = params.lambda;
      real threshold = params.threshold;
      real x         = input(irow,ibatch);
      if (x < threshold) { output(irow,ibatch) = lambda * alpha * ( std::exp(x) - 1 ); }
      else               { output(irow,ibatch) = lambda * x; }
    }


    void print_verbose() const {
      std::cout << "    alpha:     " << params.alpha     << "\n";
      std::cout << "    lambda:    " << params.lambda    << "\n";
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    void validate() const { }

  };

}


