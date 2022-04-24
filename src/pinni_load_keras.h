
#pragma once

#include "pinni.h"
#include <fstream>
#include <hdf5.h>
#include <jsoncpp/json/json.h>

namespace pinni {

  template <int MAX_LAYERS>
  inline Sequential<MAX_LAYERS> load_keras_model( std::string model_json , std::string weights_h5 ) {
    Json::Value root;
    std::ifstream file_in( model_json );
    file_in >> root;

    auto file = H5Fopen( weights_h5.c_str() , H5F_ACC_RDONLY , H5P_DEFAULT );

    pinni::Sequential<MAX_LAYERS> model;

    auto layers = root["config"]["layers"];
    int num_prev_outputs;

    for (auto &layer : layers) {
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
        yakl::Array<float,2,yakl::memHost,yakl::styleC> kernel_file("kernel_file",num_outputs,num_inputs);
        auto d_id = H5Dopen( g_id , "kernel:0" , H5P_DEFAULT );
        H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , kernel_file.data() );
        H5Dclose( d_id );
        yakl::Array<float,2,yakl::memHost,yakl::styleC> kernel("kernel",num_inputs,num_outputs);
        for (int irow = 0; irow < num_outputs; irow++) {
          for (int icol = 0; icol < num_inputs; icol++) {
            kernel(icol,irow) = kernel_file(irow,icol);
          }
        }

        // Open and read the HDF5 bias dataset
        yakl::Array<float,1,yakl::memHost,yakl::styleC> bias("bias",num_outputs);
        d_id = H5Dopen( g_id , "bias:0" , H5P_DEFAULT );
        H5Dread( d_id , H5T_NATIVE_FLOAT , H5S_ALL , H5S_ALL , H5P_DEFAULT , bias.data() );
        H5Dclose( d_id );

        // Create and add the layer
        pinni::Layer mylayer;
        mylayer.set_type       ( pinni::TYPE_DENSE         );
        mylayer.set_num_inputs ( num_inputs                );
        mylayer.set_num_outputs( num_outputs               );
        mylayer.set_kernel     ( kernel.createDeviceCopy() );
        mylayer.set_bias       ( bias.createDeviceCopy()   );
        model.add_layer( mylayer );
        num_prev_outputs = num_outputs;

      } else if (class_name == "LeakyReLU") {

        auto name = layer["config"]["name"].asString();
        int   num_inputs  = num_prev_outputs;
        int   num_outputs = num_inputs;
        float alpha       = layer["config"]["alpha"].asFloat();
        yakl::Array<float,1,yakl::memHost,yakl::styleC> params("params",1);
        params(0) = alpha;

        pinni::Layer mylayer;
        mylayer.set_type       ( pinni::TYPE_ACTIVATION_LEAKY_RELU );
        mylayer.set_num_inputs ( num_inputs                        );
        mylayer.set_num_outputs( num_outputs                       );
        mylayer.set_params     ( params.createDeviceCopy()         );
        model.add_layer( mylayer );

      } else {

        std::cout << "Error: Don't recognize the layer class_name\n";

      }
    }

    H5Fclose( file );

    return model;
  }

}


