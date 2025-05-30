#!/bin/bash

../../cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL
export KOKKOS_HOME=/home/$USER/kokkos

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DPONNI_CXX_FLAGS="-I/usr/include/hdf5/serial -O0 -g -DYAKL_DEBUG"  \
      -DYAKL_F90_FLAGS="-O0 -g"                                          \
      -DPONNI_LINK_FLAGS="" \
      -DKokkos_ENABLE_DEBUG=ON \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
      ../../..


# -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
