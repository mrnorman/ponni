#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load gcc/11.2.0 hdf5 cmake

./cmakeclean.sh

export YAKL_HOME=/ccs/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-I$OLCF_HDF5_ROOT/include -O0 -g -DYAKL_DEBUG" \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DHDF5_LINK_FLAGS="-L$OLCF_HDF5_ROOT/lib -lhdf5"        \
      ..

