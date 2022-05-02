
#pragma once

#include "YAKL.h"

namespace ponni {
  typedef float real;

  typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d     ;
  typedef yakl::Array<real      ,2,yakl::memDevice,yakl::styleC> real2d     ;

  typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
  typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;

  int constexpr UNINITIALIZED       = -1;
  int constexpr TYPE_DENSE_MATMUL   =  1;
  int constexpr TYPE_DENSE_ADD_BIAS =  2;
  int constexpr TYPE_ACT_RELU       =  3;
  int constexpr TYPE_ACT_SIGMOID    =  4;
}

#include "ponni_sequential.h"

