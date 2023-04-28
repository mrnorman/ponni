
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Selu {
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;

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
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    void init(int num_inputs, real alpha=1.67326, real lambda=1.0507, real threshold=0) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.lambda      = lambda;
      params.threshold   = threshold;
    }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        real alpha     = params_in.alpha;
        real lambda    = params_in.lambda;
        real threshold = params_in.threshold;
        real x         = input(irow,ibatch);
        if (x < threshold) { output(irow,ibatch) = lambda * alpha * ( std::exp(x) - 1 ); }
        else               { output(irow,ibatch) = lambda * x; }
      }
    }


    void print_verbose() const {
      std::cout << "    alpha:     " << params.alpha     << "\n";
      std::cout << "    lambda:    " << params.lambda    << "\n";
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    void validate() const { }

  };

}


