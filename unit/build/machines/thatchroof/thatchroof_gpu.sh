#!/bin/bash

source /usr/share/modules/init/bash
module purge

spack unload --all
spack load hdf5@1.12.2

export HDF5_DIR=/home/imn/spack/opt/spack/linux-ubuntu20.04-haswell/gcc-11.1.0/hdf5-1.12.2-yhy752tsib7atgpak6tolosnttbxlcyx

./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                         \
      -DYAKL_CUDA_FLAGS="-I$HDF5_DIR/include -O3 --use_fast_math -ccbin mpic++ -arch=sm_86" \
      -DYAKL_DEBUG=OFF                           \
      -DYAKL_PROFILE=ON                          \
      -DYAKL_AUTO_PROFILE=ON                     \
      -DYAKL_HAVE_MPI=ON                         \
      -DYAKL_F90_FLAGS="-O3"                     \
      -DHDF5_LINK_FLAGS="-L$HDF5_DIR/lib -lhdf5" \
      ..

