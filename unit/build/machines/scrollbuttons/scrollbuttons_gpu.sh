#!/bin/bash

../../cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                               \
      -DYAKL_CUDA_FLAGS="-I/usr/include/hdf5/serial -O3 --use_fast_math -res-usage -arch sm_61 -ccbin g++" \
      -DYAKL_F90_FLAGS="-O3"                                                                           \
      -DHDF5_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"        \
      ../../..

