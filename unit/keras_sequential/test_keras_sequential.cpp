
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  Kokkos::initialize( argc , argv );
  ponni::init_device_pool(128ULL*1024ULL*1024ULL); // 128 MB
  {
    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.h5>" << std::endl;
      return -1;
    }

    // This is the file with the saved tensorflow weights
    std::string fname = argv[1];

    // Create the layers that will form the model
    ponni::Matvec<float> matvec_1( ponni::load_h5_weights<2>( fname , "/dense/dense"     , "kernel:0" ) );
    ponni::Bias  <float> bias_1  ( ponni::load_h5_weights<1>( fname , "/dense/dense"     , "bias:0"   ) );
    ponni::Relu  <float> relu_1  ( 10 , 0.1 );
    ponni::Matvec<float> matvec_2( ponni::load_h5_weights<2>( fname , "/dense_1/dense_1" , "kernel:0" ) );
    ponni::Bias  <float> bias_2  ( ponni::load_h5_weights<1>( fname , "/dense_1/dense_1" , "bias:0"   ) );

    // Create an inference model to perform batched forward predictions
    auto inference = ponni::create_inference_model( matvec_1 , bias_1 , relu_1 , matvec_2 , bias_2 );
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

