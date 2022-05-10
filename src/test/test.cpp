
#include "ponni.h"
#include "ponni_load_tensorflow_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    // This is the file with the saved tensorflow weights
    std::string fname = "supercell_micro_Keras_modelwt_NORMip_NORMop1000000_Nneu10.h5";

    // Create the layers that will form the model
    ponni::Matvec matvec_1( ponni::load_tensorflow_h5_weights<2>( fname , "/dense/dense"     , "kernel:0" ) );
    ponni::Bias   bias_1  ( ponni::load_tensorflow_h5_weights<1>( fname , "/dense/dense"     , "bias:0"   ) );
    ponni::Relu   relu_1  ( 10 , 0.1 );
    ponni::Matvec matvec_2( ponni::load_tensorflow_h5_weights<2>( fname , "/dense_1/dense_1" , "kernel:0" ) );
    ponni::Bias   bias_2  ( ponni::load_tensorflow_h5_weights<1>( fname , "/dense_1/dense_1" , "bias:0"   ) );

    // Create an inference model to perform batched forward predictions
    auto inference = ponni::create_inference_model( matvec_1 , bias_1 , relu_1 , matvec_2 , bias_2 );
    inference.print_verbose();

    // Load one test sample to ensure we're getting the same outputs
    yakl::Array<float,2,yakl::memHost,yakl::styleC> inputs("inputs",12,1);
    inputs( 0,0) = 5.08810276e-01;
    inputs( 1,0) = 4.78929254e-01;
    inputs( 2,0) = 4.54260898e-01;
    inputs( 3,0) = 6.02555739e-02;
    inputs( 4,0) = 4.85583159e-02;
    inputs( 5,0) = 3.79940443e-02;
    inputs( 6,0) = 1.20564349e-04;
    inputs( 7,0) = 6.27402543e-04;
    inputs( 8,0) = 3.41872996e-03;
    inputs( 9,0) = 1.34502158e-03;
    inputs(10,0) = 4.29940776e-04;
    inputs(11,0) = 6.08758314e-06;

    // Perform a batched inference
    auto outputs = inference.batch_parallel( inputs.createDeviceCopy() );

    auto out_host = outputs.createHostCopy();

    std::cout << "Absolute difference for Output 1: " << std::abs( out_host(0,0) - 4.7658795e-01 ) << std::endl;
    std::cout << "Absolute difference for Output 2: " << std::abs( out_host(1,0) - 4.8446856e-02 ) << std::endl;
    std::cout << "Absolute difference for Output 3: " << std::abs( out_host(2,0) - 1.2472458e-03 ) << std::endl;
    std::cout << "Absolute difference for Output 4: " << std::abs( out_host(3,0) - 4.0419400e-05 ) << std::endl;

    if ( std::abs( out_host(0,0) - 4.7658795e-01 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 1 diff too large");
    if ( std::abs( out_host(1,0) - 4.8446856e-02 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 2 diff too large");
    if ( std::abs( out_host(2,0) - 1.2472458e-03 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 3 diff too large");
    if ( std::abs( out_host(3,0) - 4.0419400e-05 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 4 diff too large");

    // 4.7658795e-01 4.8446856e-02 1.2472458e-03 4.0419400e-05
  }
  yakl::finalize();
}

