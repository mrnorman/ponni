#!/bin/bash

export YAKL_HOME=/home/$USER/YAKL
PONNI_HOME=`pwd | sed 's/\(.*\/ponni\).*/\1/'`
export KOKKOS_HOME=$PONNI_HOME/unit/externals/kokkos

export CC=gcc
export CXX=g++
export FC=gfortran

unset CXXFLAGS
unset FFLAGS
unset FCFLAGS

export PONNI_DEBUG=ON
export PONNI_CXX_FLAGS="-I/usr/include/hdf5/serial -O0 -g -DYAKL_AUTO_FENCE -DYAKL_AUTO_PROFILE"
export PONNI_LINK_FLAGS=""
export PONNI_F90_FLAGS="-O0 -g"
export PONNI_BACKEND=Kokkos_ENABLE_CUDA
export PONNI_ARCH=Kokkos_ARCH_AMPERE86
