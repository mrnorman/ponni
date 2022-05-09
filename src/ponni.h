
#pragma once

#include "YAKL.h"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

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

#include "layers/ponni_Matvec.h"
#include "layers/ponni_Bias.h"
#include "layers/ponni_Relu.h"
#include "ponni_Model.h"
#include "ponni_create_model.h"

