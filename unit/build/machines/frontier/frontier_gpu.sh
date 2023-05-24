#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-amd              \
            craype-accel-amd-gfx90a \
            cray-hdf5

./cmakeclean.sh

export YAKL_HOME=/ccs/home/$USER/YAKL

unset GATOR_DISABLE

export CC=cc
export CXX=CC
export FC=ftn
unset CXXFLAGS
unset FFLAGS

export MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"

cmake -DYAKL_ARCH="HIP"         \
      -DYAKL_HIP_FLAGS="-O3 -munsafe-fp-atomics -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip -Wno-unused-result" \
      -DYAKL_F90_FLAGS="-O2"    \
      -DYAKL_DEBUG=OFF          \
      -DYAKL_PROFILE=ON         \
      -DYAKL_HAVE_MPI=ON        \
      -DHDF5_LINK_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64" \
      ..

