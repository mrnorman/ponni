
#pragma once

#include "ponni.h"

namespace ponni {

  class Bias {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real1d weights;
    };

    Params params;

    void init( real1d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights = weights;
      num_inputs     = weights.dimension[0];
      num_outputs    = weights.dimension[0];
    }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) + params.weights(irow);
    }


    void print_verbose() { }

  };



  class Bias_train {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      yakl::Array<autodiff::var<real>,1,yakl::memHost,yakl::styleC> weights;
    };

    Params params;

    void init( real1d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights = weights;
      num_inputs     = weights.dimension[0];
      num_outputs    = weights.dimension[0];
    }


    static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                   int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) + params.weights(irow);
    }


    void print_verbose() { }

  };

}


