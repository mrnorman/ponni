#!/bin/bash

./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-I/usr/include/hdf5/serial -O0 -g -DYAKL_DEBUG" \
      -DYAKL_F90_FLAGS="-O0 -g"                                         \
      -DHDF5_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"        \
      ..

