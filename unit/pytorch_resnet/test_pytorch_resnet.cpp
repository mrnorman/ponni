
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  Kokkos::initialize( argc , argv );
  ponni::init_device_pool(128ULL*1024ULL*1024ULL); // 128 MB
  {
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.h5>" << std::endl;
      return -1;
    }

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

    auto inputs   = ponni::load_h5_weights<2>( fname , "/test" , "input"  );
    auto expected = ponni::load_h5_weights<2>( fname , "/test" , "output" );
    auto outputs  = inference.forward_batch_parallel( inputs );

    auto out_host = ponni::create_host_copy(outputs);
    auto exp_host = ponni::create_host_copy(expected);

    if (out_host.extent(0) != exp_host.extent(0) || out_host.extent(1) != exp_host.extent(1)) {
      Kokkos::abort("ERROR: output dimensions do not match expected dimensions");
    }

    for (int j = 0; j < out_host.extent(1); j++) {
      for (int i = 0; i < out_host.extent(0); i++) {
        float diff = std::abs(out_host(i,j) - exp_host(i,j));
        std::cout << "Absolute difference for Output(" << i << "," << j << "): " << diff << std::endl;
        if (diff > 1.e-5f) Kokkos::abort("ERROR: output diff too large");
      }
    }
  }
  ponni::finalize_device_pool();
  Kokkos::finalize();
}

