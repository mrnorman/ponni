
#pragma once

#include "ponni.h"
#include <fstream>
#include <hdf5.h>
#include <jsoncpp/json/json.h>

namespace ponni {


  template <int MAX_LAYERS>
  inline void load_layer(Json::Value const &layer      ,
                         int &num_prev_outputs         ,
                         Sequential<MAX_LAYERS> &model ,
                         hid_t &file                   ) {
    auto class_name = layer["class_name"].asString();
    if (class_name == "InputLayer") {

      num_prev_outputs = layer["config"]["batch_input_shape"][1].asInt();

    } else if (class_name == "Dense") {

      auto name = layer["config"]["name"].asString();
      int num_inputs  = num_prev_outputs;
      int num_outputs = layer["config"]["units"].asInt();

      // Open the HDF5 group
      std::string group_name = std::string("/")+name+std::string("/")+name;
      auto g_id = H5Gopen( file , group_name.c_str() , H5P_DEFAULT );

      // Open and read the HDF5 kernel dataset
      yakl::Array<float,2,yakl::memHost,yakl::styleC> kernel("kernel_file",num_inputs,num_outputs);
      auto d_id = H5Dopen( g_id , "kernel:0" , H5P_DEFAULT );
      H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , kernel.data() );
      H5Dclose( d_id );

      // Open and read the HDF5 bias dataset
      yakl::Array<float,1,yakl::memHost,yakl::styleC> bias("bias",num_outputs);
      d_id = H5Dopen( g_id , "bias:0" , H5P_DEFAULT );
      H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , bias.data() );
      H5Dclose( d_id );

      // Create and add the matmul layer
      {
        Layer mylayer;
        mylayer.init( TYPE_DENSE_MATMUL , num_inputs , num_outputs , kernel.collapse().createDeviceCopy() );
        model.add_layer( mylayer );
      }

      // Create and add the bias layer
      {
        Layer mylayer;
        mylayer.init( TYPE_DENSE_ADD_BIAS , num_outputs , num_outputs , bias.createDeviceCopy() );
        model.add_layer( mylayer );
      }

      num_prev_outputs = num_outputs;

    } else if (class_name == "LeakyReLU") {

      auto name = layer["config"]["name"].asString();
      int   num_inputs  = num_prev_outputs;
      int   num_outputs = num_inputs;
      float alpha       = layer["config"]["alpha"].asFloat();
      yakl::Array<float,1,yakl::memHost,yakl::styleC> params("params",3);
      params(0) = std::numeric_limits<real>::max();
      params(1) = alpha;
      params(2) = 0;
      Layer mylayer;
      mylayer.init( TYPE_ACT_RELU , num_inputs , num_outputs , params.createDeviceCopy() );
      model.add_layer( mylayer );

    } else if (class_name == "ReLU") {

      auto name = layer["config"]["name"].asString();
      int   num_inputs  = num_prev_outputs;
      int   num_outputs = num_inputs;
      yakl::Array<float,1,yakl::memHost,yakl::styleC> params("params",3);
      real max_value      = layer["config"]["max_value"     ].asFloat();
      real negative_slope = layer["config"]["negative_slope"].asFloat();
      real threshold      = layer["config"]["threshold"     ].asFloat();
      if (max_value == 0) max_value = std::numeric_limits<real>::max();
      params(0) = max_value;
      params(1) = negative_slope;
      params(2) = threshold;
      Layer mylayer;
      mylayer.init( TYPE_ACT_RELU , num_inputs , num_outputs , params.createDeviceCopy() );
      model.add_layer( mylayer );

    } else {

      std::cout << "Error: Don't recognize the layer class_name\n";

    }
  }



  template <int MAX_LAYERS>
  inline Sequential<MAX_LAYERS> load_keras_model( std::string model_json , std::string weights_h5 ) {
    Json::Value root;
    std::ifstream file_in( model_json );
    file_in >> root;

    auto file = H5Fopen( weights_h5.c_str() , H5F_ACC_RDONLY , H5P_DEFAULT );

    Sequential<MAX_LAYERS> model;

    auto layers = root["config"]["layers"];
    int num_prev_outputs;

    for (auto &layer : layers) { load_layer( layer , num_prev_outputs , model , file ); }

    H5Fclose( file );

    return model;
  }

}


