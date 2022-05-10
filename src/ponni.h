
#pragma once

#include "YAKL.h"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

namespace ponni {
  typedef float                    real;
  typedef autodiff::Variable<real> real_trainable;

  typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d     ;
  typedef yakl::Array<real      ,2,yakl::memDevice,yakl::styleC> real2d     ;
  typedef yakl::Array<real      ,3,yakl::memDevice,yakl::styleC> real3d     ;

  typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
  typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;
  typedef yakl::Array<real const,3,yakl::memDevice,yakl::styleC> realConst3d;

  typedef yakl::Array<real      ,1,yakl::memHost,yakl::styleC> realHost1d     ;
  typedef yakl::Array<real      ,2,yakl::memHost,yakl::styleC> realHost2d     ;
  typedef yakl::Array<real      ,3,yakl::memHost,yakl::styleC> realHost3d     ;

  typedef yakl::Array<real const,1,yakl::memHost,yakl::styleC> realConstHost1d;
  typedef yakl::Array<real const,2,yakl::memHost,yakl::styleC> realConstHost2d;
  typedef yakl::Array<real const,3,yakl::memHost,yakl::styleC> realConstHost3d;

  typedef yakl::Array<real_trainable      ,1,yakl::memHost,yakl::styleC> real1d_trainable     ;
  typedef yakl::Array<real_trainable      ,2,yakl::memHost,yakl::styleC> real2d_trainable     ;
  typedef yakl::Array<real_trainable      ,3,yakl::memHost,yakl::styleC> real3d_trainable     ;

  typedef yakl::Array<real_trainable const,1,yakl::memHost,yakl::styleC> realConst1d_trainable;
  typedef yakl::Array<real_trainable const,2,yakl::memHost,yakl::styleC> realConst2d_trainable;
  typedef yakl::Array<real_trainable const,3,yakl::memHost,yakl::styleC> realConst3d_trainable;
}

#include "layers/ponni_Matvec.h"
#include "layers/ponni_Bias.h"
#include "layers/ponni_Relu.h"
#include "ponni_Inference.h"
#include "ponni_create_model.h"

