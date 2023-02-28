#!/bin/bash

source /usr/share/modules/init/bash
module purge
# cuda-10.2.89-gcc-11.1.0-52kbsh3
# cuda-11.1.0-gcc-11.1.0-zkj7zay 
# cuda-11.2.0-gcc-11.1.0-u7lo6pf
# cuda-11.3.0-gcc-11.1.0-62t6da7
# cuda-11.4.0-gcc-11.1.0-xcs7jho
# cuda-11.5.0-gcc-11.1.0-pm6vauh
# cuda-11.6.0-gcc-11.1.0-h7khsvh
# cuda-11.7.0-gcc-11.1.0-zbtc2fk
module load cuda-11.6.0-gcc-11.1.0-h7khsvh
./cmakeclean.sh

export YAKL_HOME=/home/$USER/YAKL

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA" \
      -DYAKL_CUDA_FLAGS="-I/usr/include/hdf5/serial -O3 -ccbin mpic++ -arch=sm_86" \
      -DYAKL_F90_FLAGS="-O3"                                               \
      -DHDF5_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"        \
      ..

