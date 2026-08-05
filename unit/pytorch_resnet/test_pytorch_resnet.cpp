
#include "ponni.h"

int main( int argc , char **argv ) {
  Kokkos::initialize( argc , argv );
  {
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Silu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.ponni>" << std::endl;
      return -1;
    }

    std::string fname = argv[1];
    ponni::PonniFile file;
    std::string error;
    if (!file.load(fname,&error)) throw std::runtime_error(error);

    // Create layers & load weights
    ponni::Matvec<float> matvec_1(ponni::load_ponni_tensor<2>(file,"fc1.weight"));
    ponni::Bias  <float> bias_1  (ponni::load_ponni_tensor<1>(file,"fc1.bias"));
    ponni::Matvec<float> matvec_2(ponni::load_ponni_tensor<2>(file,"fc2.weight"));
    ponni::Bias  <float> bias_2  (ponni::load_ponni_tensor<1>(file,"fc2.bias"));
    ponni::Matvec<float> matvec_3(ponni::load_ponni_tensor<2>(file,"fc3.weight"));
    ponni::Bias  <float> bias_3  (ponni::load_ponni_tensor<1>(file,"fc3.bias"));
    ponni::Matvec<float> matvec_4(ponni::load_ponni_tensor<2>(file,"fc4.weight"));
    ponni::Bias  <float> bias_4  (ponni::load_ponni_tensor<1>(file,"fc4.bias"));
    ponni::Matvec<float> matvec_5(ponni::load_ponni_tensor<2>(file,"fc5.weight"));
    ponni::Bias  <float> bias_5  (ponni::load_ponni_tensor<1>(file,"fc5.bias"));

    // Create an inference model to perform batched forward predictions
    auto inference = create_inference_model( matvec_1                       ,
                                             bias_1                         ,
                                             Silu        <float>( 5 )                            ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_2                       ,
                                             bias_2                         ,
                                             Silu        <float>( 5 )                            ,
                                             Binop_Add <0,float>( 5 )       ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_3                       ,
                                             bias_3                         ,
                                             Silu        <float>( 5 )                            ,
                                             Binop_Add <0,float>( 5 )       ,
                                             Save_State<0,float>( 5 )       ,
                                             matvec_4                       ,
                                             bias_4                         ,
                                             Silu        <float>( 5 )                            ,
                                             Binop_Add <0,float>( 5 )       ,
                                             matvec_5                       ,
                                             bias_5                         );
                                                   
    inference.validate();
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
