
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  Kokkos::initialize( argc , argv );
  {
    typedef Kokkos::View<float * ,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space> real1d;
    typedef Kokkos::View<float **,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space> real2d;
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::LeakyRelu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    // Create layers & load weights
    int num_layers  = 8;
    int num_batches = 1024*1024*16;
    int num_runs    = 10;

    real2d weights_1("matvec_1",num_layers,num_layers) ;
    real1d weights_2("bias_1",num_layers)              ;
    real2d weights_3("matvec_1",num_layers,num_layers) ;
    real1d weights_4("bias_1",num_layers)              ;
    Kokkos::deep_copy( weights_1 , 1. );
    Kokkos::deep_copy( weights_2 , 1. );
    Kokkos::deep_copy( weights_3 , 1. );
    Kokkos::deep_copy( weights_4 , 1. );

    // Create an inference model to perform batched forward predictions
    auto inference = create_inference_model( Matvec<float>( weights_1 )        ,
                                             Bias  <float>( weights_2 )        ,
                                             LeakyRelu<float>(num_layers, 0.1) ,
                                             Matvec<float>( weights_3 )        ,
                                             Bias  <float>( weights_4 )        ,
                                             LeakyRelu<float>(num_layers, 0.1) );
                                              
    inference.validate();
    inference.print();

    // Perform a batched inference
    real2d outputs;
    for (int i=0; i < num_runs; i++) {
      real2d inputs("input",num_layers,num_batches);
      Kokkos::deep_copy( inputs , 0.1 );
      outputs = inference.forward_batch_parallel( inputs );
    }

    double val = ponni::intrinsics::sum( outputs ) / num_layers / num_batches;
    std::cout << std::scientific << std::abs(val-15.4) << std::endl;
    if (std::abs(val-15.4) > 1.e-3) {
      throw std::runtime_error("Error: Batched inference failed");
    }
  }
  Kokkos::finalize();
}
