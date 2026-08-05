
#include "ponni.h"

int main( int argc , char **argv ) {
  Kokkos::initialize( argc , argv );
  {
    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.ponni>" << std::endl;
      return -1;
    }

    std::string fname = argv[1];
    ponni::PonniFile file;
    std::string error;
    if (!file.load(fname,&error)) throw std::runtime_error(error);

    // Create the layers that will form the model
    ponni::Matvec<float> matvec_1(ponni::load_ponni_tensor<2>(file,"dense.kernel"));
    ponni::Bias  <float> bias_1  (ponni::load_ponni_tensor<1>(file,"dense.bias"));
    ponni::Tanh<float> act_1(10);
    ponni::Matvec<float> matvec_2(ponni::load_ponni_tensor<2>(file,"dense_1.kernel"));
    ponni::Bias  <float> bias_2  (ponni::load_ponni_tensor<1>(file,"dense_1.bias"));

    // Create an inference model to perform batched forward predictions
    auto inference = ponni::create_inference_model( matvec_1 , bias_1 , act_1 , matvec_2 , bias_2 );
    inference.print();

    auto inputs   = ponni::load_ponni_tensor<2>(file,"test.input");
    auto expected = ponni::load_ponni_tensor<2>(file,"test.output");
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
  Kokkos::finalize();
}
