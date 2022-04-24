
#pragma once

namespace pinni {
  typedef float real;
  typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d     ;
  typedef yakl::Array<real      ,2,yakl::memDevice,yakl::styleC> real2d     ;

  typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
  typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;
}

#include "pinni_sequential.h"

