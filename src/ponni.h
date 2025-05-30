
#pragma once

#include "YAKL.h"

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
#include "initializers/ponni_Initializer_Random_Uniform.h"
#include "layers/ponni_Matvec.h"
#include "layers/ponni_Bias.h"
#include "layers/ponni_Relu.h"
#include "layers/ponni_Save_State.h"
#include "layers/ponni_Binop_Add.h"
#include "layers/ponni_Binop_Concatenate.h"
#include "ponni_Inference.h"
#include "ponni_create_model.h"



