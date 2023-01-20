
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    using ponni::real1d;
    using ponni::real2d;
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    // Create layers & load weights
    int num_layers  = 8;
    int num_batches = 1024*1024*16;
    int num_runs    = 10;

    // Create an inference model to perform batched forward predictions
    auto inference = create_inference_model( Matvec( real2d("matvec_1",num_layers,num_layers) = 1 ) ,
                                             Bias  ( real1d("bias_1",num_layers) = 1 )              ,
                                             Relu  ( num_layers , 0.1 )                             ,
                                             Matvec( real2d("matvec_1",num_layers,num_layers) = 1 ) ,
                                             Bias  ( real1d("bias_1",num_layers) = 1 )              ,
                                             Relu  ( num_layers , 0.1 )                             );
                                              
    inference.validate();
    inference.print();

    // Perform a batched inference
    real2d outputs;
    for (int i=0; i < num_runs; i++) {
      outputs = inference.batch_parallel( real2d("input",num_layers,num_batches) = 0.1 );
    }

    std::cout << yakl::intrinsics::sum( outputs ) / num_layers / num_batches << std::endl;
  }
  yakl::finalize();
}

