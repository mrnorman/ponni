
#pragma once

#include "YAKL.h"

namespace ponni {
  typedef float real;

  typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d;
  typedef yakl::Array<real      ,2,yakl::memDevice,yakl::styleC> real2d;
  typedef yakl::Array<real      ,3,yakl::memDevice,yakl::styleC> real3d;
  typedef yakl::Array<real      ,4,yakl::memDevice,yakl::styleC> real4d;
  typedef yakl::Array<real      ,5,yakl::memDevice,yakl::styleC> real5d;

  typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
  typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;
  typedef yakl::Array<real const,3,yakl::memDevice,yakl::styleC> realConst3d;
  typedef yakl::Array<real const,4,yakl::memDevice,yakl::styleC> realConst4d;
  typedef yakl::Array<real const,5,yakl::memDevice,yakl::styleC> realConst5d;

  typedef yakl::Array<real      ,1,yakl::memHost,yakl::styleC> realHost1d;
  typedef yakl::Array<real      ,2,yakl::memHost,yakl::styleC> realHost2d;
  typedef yakl::Array<real      ,3,yakl::memHost,yakl::styleC> realHost3d;
  typedef yakl::Array<real      ,4,yakl::memHost,yakl::styleC> realHost4d;
  typedef yakl::Array<real      ,5,yakl::memHost,yakl::styleC> realHost5d;

  typedef yakl::Array<real const,1,yakl::memHost,yakl::styleC> realConstHost1d;
  typedef yakl::Array<real const,2,yakl::memHost,yakl::styleC> realConstHost2d;
  typedef yakl::Array<real const,3,yakl::memHost,yakl::styleC> realConstHost3d;
  typedef yakl::Array<real const,4,yakl::memHost,yakl::styleC> realConstHost4d;
  typedef yakl::Array<real const,5,yakl::memHost,yakl::styleC> realConstHost5d;
}

#include "layers/ponni_Matvec.h"
#include "layers/ponni_Bias.h"
#include "layers/ponni_Relu.h"
#include "layers/ponni_Binary_Step.h"
#include "layers/ponni_Sigmoid.h"
#include "layers/ponni_Tanh.h"
#include "layers/ponni_Gelu.h"
#include "layers/ponni_Softplus.h"
#include "layers/ponni_Selu.h"
#include "layers/ponni_Prelu.h"
#include "layers/ponni_Silu.h"
#include "layers/ponni_Gaussian.h"
#include "ponni_Inference.h"
#include "ponni_create_model.h"

