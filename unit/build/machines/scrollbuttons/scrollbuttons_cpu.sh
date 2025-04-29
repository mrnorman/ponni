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

cmake -DPONNI_CXX_FLAGS="-I/usr/include/hdf5/serial -O3"  \
      -DYAKL_F90_FLAGS="-O3"                                          \
      -DPONNI_LINK_FLAGS="" \
      -DKokkos_ARCH_NATIVE=ON \
      -DKokkos_ENABLE_DEBUG=OFF \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
      ../../..


# -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
