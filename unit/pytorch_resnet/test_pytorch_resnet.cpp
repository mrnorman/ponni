
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    std::string fname = argv[1];

    // Create layers & load weights
    bool transpose = true;
    ponni::Matvec<float> matvec_1( ponni::load_h5_weights<2>( fname , "/" , "0.0.0.0.1.weight"            , transpose ) );
    ponni::Bias  <float> bias_1  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.0.0.1.bias"              , transpose ) );
    ponni::Matvec<float> matvec_2( ponni::load_h5_weights<2>( fname , "/" , "0.0.0.2.sequential.0.weight" , transpose ) );
    ponni::Bias  <float> bias_2  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.0.2.sequential.0.bias"   , transpose ) );
    ponni::Matvec<float> matvec_3( ponni::load_h5_weights<2>( fname , "/" , "0.0.2.sequential.0.weight"   , transpose ) );
    ponni::Bias  <float> bias_3  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.2.sequential.0.bias"     , transpose ) );
    ponni::Matvec<float> matvec_4( ponni::load_h5_weights<2>( fname , "/" , "0.2.sequential.0.weight"     , transpose ) );
    ponni::Bias  <float> bias_4  ( ponni::load_h5_weights<1>( fname , "/" , "0.2.sequential.0.bias"       , transpose ) );
    ponni::Matvec<float> matvec_5( ponni::load_h5_weights<2>( fname , "/" , "2.weight"                    , transpose ) );
    ponni::Bias  <float> bias_5  ( ponni::load_h5_weights<1>( fname , "/" , "2.bias"                      , transpose ) );

    // Create an inference model to perform batched forward predictions
    auto inference = create_inference_model( matvec_1                       ,
                                             bias_1                         ,
                                             Relu        <float>( 5 , 0.1 ) ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_2                       ,
                                             bias_2                         ,
                                             Relu        <float>( 5 , 0.1 ) ,
                                             Binop_Add <0,float>( 5 )       ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_3                       ,
                                             bias_3                         ,
                                             Relu        <float>( 5 , 0.1 ) ,
                                             Binop_Add <0,float>( 5 )       ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_4                       ,
                                             bias_4                         ,
                                             Relu        <float>( 5 , 0.1 ) ,
                                             Binop_Add <0,float>( 5 )       ,
                                             matvec_5                       ,
                                             bias_5                         );
                                                   
    inference.validate();
    inference.print();

    yakl::Array<float,2,yakl::memHost,yakl::styleC> inputs("inputs",12,1);
    inputs( 0,0) = 5.0881004e-01;
    inputs( 1,0) = 4.7892904e-01;
    inputs( 2,0) = 4.5426106e-01;
    inputs( 3,0) = 6.0255572e-02;
    inputs( 4,0) = 4.8558313e-02;
    inputs( 5,0) = 3.7994046e-02;
    inputs( 6,0) = 1.2056435e-04;
    inputs( 7,0) = 6.2740257e-04;
    inputs( 8,0) = 3.4187301e-03;
    inputs( 9,0) = 1.3450217e-03;
    inputs(10,0) = 4.2994076e-04;
    inputs(11,0) = 6.0875832e-06;

    // Perform a batched inference
    auto outputs = inference.forward_batch_parallel( inputs.createDeviceCopy() );

    auto out_host = outputs.createHostCopy();

    std::cout << "Absolute difference for Output 1: " << std::abs( out_host(0,0) - ( 4.75395530e-01) ) << std::endl;
    std::cout << "Absolute difference for Output 2: " << std::abs( out_host(1,0) - ( 4.74427342e-02) ) << std::endl;
    std::cout << "Absolute difference for Output 3: " << std::abs( out_host(2,0) - ( 4.08545136e-04) ) << std::endl;
    std::cout << "Absolute difference for Output 4: " << std::abs( out_host(3,0) - (-1.02701783e-03) ) << std::endl;

    if ( std::abs( out_host(0,0) - ( 4.75395530e-01) ) > 1.e-6 ) Kokkos::abort("ERROR Output 1 diff too large");
    if ( std::abs( out_host(1,0) - ( 4.74427342e-02) ) > 1.e-6 ) Kokkos::abort("ERROR Output 2 diff too large");
    if ( std::abs( out_host(2,0) - ( 4.08545136e-04) ) > 1.e-6 ) Kokkos::abort("ERROR Output 3 diff too large");
    if ( std::abs( out_host(3,0) - (-1.02701783e-03) ) > 1.e-6 ) Kokkos::abort("ERROR Output 4 diff too large");
  }
  yakl::finalize();
}

