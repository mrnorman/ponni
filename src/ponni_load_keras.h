
#pragma once

#include "ponni.h"
#include <fstream>
#include <hdf5.h>
#include <jsoncpp/json/json.h>

namespace ponni {

  template <int N>
  inline void load_weights(yakl::Array<float,N,yakl::memDevice,yakl::styleC> const &weights ,
                           std::string file_name   ,
                           std::string group_name  ,
                           std::string dataset_name ) {
    auto f_id = H5Fopen( file_name.c_str() , H5F_ACC_RDONLY , H5P_DEFAULT );
    auto g_id = H5Gopen( f_id , group_name  .c_str() , H5P_DEFAULT );
    auto d_id = H5Dopen( g_id , dataset_name.c_str() , H5P_DEFAULT );
    yakl::Array<float,N,yakl::memHost,yakl::styleC> weights_host = weights.createHostObject();
    H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , weights_host.data() );
    weights_host.deep_copy_to(weights);
    H5Dclose( d_id );
    H5Gclose( g_id );
    H5Fclose( f_id );
  }

}


