
#pragma once

#include "YAKL.h"

namespace ponni {

  class Act_leakyrelu {
  public:

    static void init( int num_inputs , int num_outputs , realConst1d weights , bool &overwrite_input ) {
      overwrite_input = false;
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Act_leakyrelu weights vector not initialized");
      if ( num_inputs != num_outputs ) yakl::yakl_throw("ERROR: Act_leakyrelu num_inputs != num_outputs");
      if ( weights.totElems() != 1 ) yakl::yakl_throw("ERROR: Act_leakyrelu defined more than one param");
    }


    YAKL_INLINE static void apply_1(realConst1d weights, int num_inputs, int num_outputs,
                                    realConst2d input, real2d const &output, int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) >= 0 ? input(irow,ibatch) : weights(0) * input(irow,ibatch);
    }


    static void print_verbose(realConst1d weights, int num_inputs , int num_outputs ) {
      std::cout << "    slope: " << std::setw(12) << weights.createHostCopy()(0) << "\n";
    }

  };

}


