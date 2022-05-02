
#pragma once

#include "YAKL.h"

namespace ponni {

  class Act_relu {
  public:

    static void init( int num_inputs , int num_outputs , realConst1d weights , bool &overwrite_input ) {
      overwrite_input = false;
      if ( num_inputs != num_outputs ) yakl::yakl_throw("ERROR: Act_relu num_inputs != num_outputs");
    }


    YAKL_INLINE static void apply_1(realConst1d weights, int num_inputs, int num_outputs,
                                    realConst2d input, real2d const &output, int ibatch, int irow) {
      output(irow,ibatch) = std::max( input(irow,ibatch) , static_cast<real>(0.) );
    }


    static void print_verbose(realConst1d weights, int num_inputs , int num_outputs ) { }

  };

}


