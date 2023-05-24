#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load hdf5-1.12.2-gcc-11.1.0-yhy752t

spack unload --all

export HDF5_DIR=`which h5cc | xargs dirname`/..

./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                         \
      -DYAKL_CUDA_FLAGS="-I$HDF5_DIR/include -O2 -ccbin mpic++ -arch=sm_86" \
      -DYAKL_DEBUG=OFF                           \
      -DYAKL_PROFILE=ON                          \
      -DYAKL_AUTO_PROFILE=OFF                    \
      -DYAKL_AUTO_FENCE=OFF                      \
      -DYAKL_HAVE_MPI=ON                         \
      -DYAKL_F90_FLAGS="-O3"                     \
      -DHDF5_LINK_FLAGS="-L$HDF5_DIR/lib -lhdf5" \
      ..

