#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load gcc hdf5 cmake cuda

./cmakeclean.sh

export YAKL_HOME=/ccs/home/$USER/YAKL

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

export MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"

cmake -DYAKL_ARCH="CUDA" \
      -DYAKL_CUDA_FLAGS="-arch sm_70 -I$OLCF_HDF5_ROOT/include -O3 --use_fast_math -res-usage" \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DHDF5_LINK_FLAGS="-L$OLCF_HDF5_ROOT/lib -lhdf5"        \
      ..

