
#pragma once
// Included by ponni.h

namespace ponni {

  class Bias {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real1d weights;
    };

    Params params;

    Bias() {}
    Bias(real1d const &weights) { init(weights); }


    void init( real1d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[0];
      params.weights     = weights;
    }


    char const * get_label      () const { return "Bias"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE void compute_one_output(realConst2d input, real2d const &output,
                                        int ibatch, int irow) const {
      output(irow,ibatch) = input(irow,ibatch) + params.weights(irow);
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      auto bias_host = params.weights.createHostCopy();
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      " << std::setw(12) << bias_host(irow) << "\n";
      }
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


