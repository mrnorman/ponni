
#pragma once
// Included by ponni.h

namespace ponni {

  class Matvec {
  public:
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real2d weights;
    };

    Params params;

    Matvec()  = default;
    ~Matvec() = default;
    Matvec(real2d const &weights) { init(weights); }


    void init( real2d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[1];
      params.weights     = weights;
    }


    char const * get_label         () const { return "Matvec"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE void compute_all_outputs(real2d const &input, real2d const &output, int ibatch) const {
      for (int irow = 0; irow < params.num_outputs; irow++) {
        auto &weights = params.weights;
        real tmp = 0;
        for (int k=0; k < params.num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
        output(irow,ibatch) = tmp;
      }
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      auto kernel_host = params.weights.createHostCopy();
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      ";
        for (int icol=0; icol < params.num_inputs; icol++) {
          std::cout << std::setw(12) << std::setprecision(4) << kernel_host(icol,irow) << "  ";
        }
        std::cout << "\n";
      }
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


