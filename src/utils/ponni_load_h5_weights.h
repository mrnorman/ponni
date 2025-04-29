
#pragma once

#include "ponni.h"
#include <fstream>
#include <hdf5.h>

namespace ponni {

  template <int N>
  inline yakl::Array<float,N,yakl::memDevice,yakl::styleC> 
  load_h5_weights( std::string file_name    ,
                   std::string group_name   ,
                   std::string dataset_name ,
                   bool transpose = false   ) {
    static_assert( N >= 1 && N <= 4 , "ERROR: load_h5_weights array rank must be betwee 1 and 4" );
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    typedef typename yakl::Array<float,N,yakl::memHost,yakl::styleC> arr_type;
    auto f_id = H5Fopen( file_name.c_str() , H5F_ACC_RDONLY , H5P_DEFAULT );
    auto g_id = H5Gopen( f_id , group_name  .c_str() , H5P_DEFAULT );
    auto d_id = H5Dopen( g_id , dataset_name.c_str() , H5P_DEFAULT );
    auto dspace = H5Dget_space(d_id);
    hsize_t dims[N];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    arr_type weights_host;
    if      constexpr (N == 1) { weights_host = arr_type("weights_host",dims[0]);                         }
    else if constexpr (N == 2) { weights_host = arr_type("weights_host",dims[0],dims[1]);                 }
    else if constexpr (N == 3) { weights_host = arr_type("weights_host",dims[0],dims[1],dims[2]);         }
    else if constexpr (N == 4) { weights_host = arr_type("weights_host",dims[0],dims[1],dims[2],dims[3]); }
    H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , weights_host.data() );
    H5Dclose( d_id );
    H5Gclose( g_id );
    H5Fclose( f_id );

    if ( N == 1 || (! transpose) ) {
      return weights_host.createDeviceCopy();
    } else {
      auto weights_tmp = weights_host.createDeviceCopy();
      if        constexpr (N == 2) {
        yakl::Array<float,N,yakl::memDevice,yakl::styleC> weights_transposed("weights_transposed",dims[1],dims[0]);
        parallel_for( SimpleBounds<N>(dims[0],dims[1]) , KOKKOS_LAMBDA (int i0, int i1) {
          weights_transposed(i1,i0) = weights_tmp(i0,i1);
        });
        return weights_transposed;
      } else if constexpr (N == 3) {
        yakl::Array<float,N,yakl::memDevice,yakl::styleC> weights_transposed("weights_transposed",dims[2],dims[1],dims[0]);
        parallel_for( SimpleBounds<N>(dims[0],dims[1],dims[2]) , KOKKOS_LAMBDA (int i0, int i1, int i2) {
          weights_transposed(i2,i1,i0) = weights_tmp(i0,i1,i2);
        });
        return weights_transposed;
      } else if constexpr (N == 4) {
        yakl::Array<float,N,yakl::memDevice,yakl::styleC> weights_transposed("weights_transposed",dims[3],dims[2],dims[1],dims[0]);
        parallel_for( SimpleBounds<N>(dims[0],dims[1],dims[2],dims[3]) , KOKKOS_LAMBDA (int i0, int i1, int i2, int i3) {
          weights_transposed(i3,i2,i1,i0) = weights_tmp(i0,i1,i2,i3);
        });
        return weights_transposed;
      }
      Kokkos::abort("ERROR: Shouldn't have reached here");
      return yakl::Array<float,N,yakl::memDevice,yakl::styleC>();
    }
  }

}


