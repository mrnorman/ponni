
#pragma once

#include "YAKL.h"

namespace ponni {

  class Dense_matmul {
  public:

    static void init( int num_inputs , int num_outputs , realConst1d weights , bool &overwrite_input ) {
      overwrite_input = true;
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Dense_matmul weights matrix not initialized");
      if ( weights.totElems() != num_inputs * num_outputs ) {
        yakl::yakl_throw("ERROR: Dense_matmul total weights != num_inputs * num_outputs");
      }
    }


    YAKL_INLINE static void apply_1(realConst1d weights, int num_inputs, int num_outputs,
                                    realConst2d input, real2d const &output, int ibatch, int irow) {
      real tmp = 0;
      for (int k=0; k < num_inputs; k++) { tmp += weights(k*num_outputs+irow) * input(k,ibatch); }
      output(irow,ibatch) = tmp;
    }


    static void print_verbose(realConst1d weights, int num_inputs , int num_outputs ) {
      std::cout << "    matrix:\n";
      auto weights_host = weights.createHostCopy();
      for (int irow=0; irow < num_outputs; irow++) {
        std::cout << "      ";
        for (int icol=0; icol < num_inputs; icol++) {
          std::cout << std::setw(12) << weights_host(icol*num_outputs+irow) << "  ";
        }
        std::cout << "\n";
      }
    }

  };

}


