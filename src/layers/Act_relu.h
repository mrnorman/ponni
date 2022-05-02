
#pragma once

#include "YAKL.h"

namespace ponni {

  class Act_relu {
  public:

    static void init( int num_inputs , int num_outputs , realConst1d weights , bool &overwrite_input ) {
      overwrite_input = false;
      if ( num_inputs != num_outputs ) yakl::yakl_throw("ERROR: Act_relu num_inputs != num_outputs");
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Act_relu weights vector not initialized");
      if ( weights.totElems() != 3 ) yakl::yakl_throw("ERROR: Act_relu must have exactly 3 parameters");
    }


    YAKL_INLINE static void apply_1(realConst1d weights, int num_inputs, int num_outputs,
                                    realConst2d input, real2d const &output, int ibatch, int irow) {
      real max_value      = weights(0);
      real negative_slope = weights(1);
      real threshold      = weights(2);
      real x              = input(irow,ibatch);
      real f_x;
      if      (x >= max_value) { f_x = max_value;                        }
      else if (x >= threshold) { f_x = x;                                }
      else                     { f_x = negative_slope * (x - threshold); }
      output(irow,ibatch) = f_x;
    }


    static void print_verbose(realConst1d weights, int num_inputs , int num_outputs ) { }

  };

}


