
#pragma once

#include "YAKL.h"

namespace ponni {

  class Dense_bias {
  public:

    static void init( int num_inputs , int num_outputs , realConst1d weights , bool &overwrite_input ) {
      overwrite_input = false;
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Dense_bias weights vector not initialized");
      if ( num_inputs != num_outputs ) yakl::yakl_throw("ERROR: Dense_bias num_inputs != num_outputs");
      if ( weights.totElems() != num_inputs ) yakl::yakl_throw("ERROR: Dense_bias total weights != num_inputs");
    }


    YAKL_INLINE static void apply_1(realConst1d weights, int num_inputs, int num_outputs,
                                    realConst2d input, real2d const &output, int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) + weights(irow);
    }


    static void print_verbose(realConst1d weights, int num_inputs , int num_outputs ) {
      std::cout << "    vector:\n";
      auto weights_host = weights.createHostCopy();
      for (int i=0; i < num_inputs; i++) {
        std::cout << "      " << std::setw(12) << weights_host(i) << "\n";
      }
    }

  };

}


