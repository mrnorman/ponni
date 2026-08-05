
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "ponni_kokkos_utils.h"
#include <iomanip>

namespace ponni {
  inline void debug_print( char const * file , int line ) {
    std::cout << "*** DEBUG: " << file << ": " << line << std::endl;
  }
  template <class T> inline void debug_print_val( T var , char const * file , int line , char const * varname ) {
    std::cout << "*** DEBUG: " << file << ": " << line << ": " << varname << "  -->  " << var << std::endl;
  }
}

#define PONNI_DEBUG_PRINT() { ponni::debug_print(__FILE__,__LINE__); }
#define PONNI_DEBUG_PRINT_VAL(var) { ponni::debug_print_val((var),__FILE__,__LINE__,#var); }

#include <fstream>
#include <random>
#include <algorithm>
#include <type_traits>
#include "initializers/ponni_initializer.h"
#include "ponni_LayerTraits.h"
#include "layers/ponni_Matvec.h"
#include "layers/ponni_Bias.h"
#include "layers/ponni_Relu.h"
#include "layers/ponni_LeakyRelu.h"
#include "layers/ponni_Elu.h"
#include "layers/ponni_Selu.h"
#include "layers/ponni_Gelu.h"
#include "layers/ponni_Silu.h"
#include "layers/ponni_Sigmoid.h"
#include "layers/ponni_Tanh.h"
#include "layers/ponni_Softmax.h"
#include "layers/ponni_LogSoftmax.h"
#include "layers/ponni_Softplus.h"
#include "layers/ponni_HardSigmoid.h"
#include "layers/ponni_HardSwish.h"
#include "layers/ponni_Mish.h"
#include "layers/ponni_LayerNorm.h"
#include "layers/ponni_MinMaxNorm.h"
#include "layers/ponni_Save_State.h"
#include "layers/ponni_Binop_Add.h"
#include "layers/ponni_Binop_Concatenate.h"
#include "layers/ponni_Binop_Projection_Add.h"
#include "ponni_Inference.h"
#include "ponni_create_model.h"
