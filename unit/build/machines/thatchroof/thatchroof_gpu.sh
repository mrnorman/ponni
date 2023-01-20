#!/bin/bash

./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA" \
      -DYAKL_CUDA_FLAGS="-I/usr/include/hdf5/serial -O3 -DYAKL_AUTO_FENCE -arch=sm_86" \
      -DYAKL_F90_FLAGS="-O3"                                               \
      -DHDF5_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"        \
      ..

