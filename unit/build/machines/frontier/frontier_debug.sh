#!/bin/bash

$MODULESHOME/init/bash
module reset
module load PrgEnv-amd cray-hdf5

../../cmakeclean.sh

export YAKL_HOME=/ccs/home/$USER/YAKL
export KOKKOS_HOME=/ccs/home/$USER/kokkos

unset GATOR_DISABLE
export CRAYPE_LINK_TYPE=dynamic

export CC=cc
export CXX=CC
export FC=ftn
unset CXXFLAGS
unset FFLAGS

cmake -DPONNI_CXX_FLAGS="-I/usr/include/hdf5/serial -O0 -g -DYAKL_DEBUG"  \
      -DYAKL_F90_FLAGS="-O0 -g"                                           \
      -DPONNI_LINK_FLAGS="" \
      -DKokkos_ENABLE_DEBUG=ON \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
      ../../..


# -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
