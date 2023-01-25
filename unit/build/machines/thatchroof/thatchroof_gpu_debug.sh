#!/bin/bash

source /usr/share/modules/init/bash
module purge
# cuda-11.5.0-gcc-11.1.0-pm6vauh
# cuda-11.6.0-gcc-11.1.0-h7khsvh
# cuda-11.7.0-gcc-11.1.0-zbtc2fk
module load cuda-11.7.0-gcc-11.1.0-zbtc2fk
./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA" \
      -DYAKL_CUDA_FLAGS="-I/usr/include/hdf5/serial -O0 -g -arch=sm_86 -DYAKL_DEBUG" \
      -DYAKL_F90_FLAGS="-O0 -g"                                         \
      -DHDF5_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"        \
      ..

