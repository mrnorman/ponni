
#pragma once

#include "YAKL.h"

namespace pinni {
  typedef float real;
  typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d     ;
  typedef yakl::Array<real      ,2,yakl::memDevice,yakl::styleC> real2d     ;

  typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
  typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;

  int constexpr UNINITIALIZED              = -1;
  int constexpr TYPE_DENSE                 =  1;
  int constexpr TYPE_ACTIVATION_RELU       =  2;
  int constexpr TYPE_ACTIVATION_LEAKY_RELU =  3;
}

#include "pinni_sequential.h"

