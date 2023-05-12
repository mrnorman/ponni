
#include "ponni.h"

int main( int argc , char **argv ) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    typedef float real;
    typedef yakl::Array<real,1,yakl::memDevice> real1d;
    typedef yakl::Array<real,2,yakl::memDevice> real2d;
    typedef yakl::Array<real,3,yakl::memDevice> real3d;

    // Create training data
    int training_size = 1024;
    real1d training_inputs ("training_inputs" ,training_size);
    real1d training_outputs("training_outputs",training_size);
    parallel_for( YAKL_AUTO_LABEL() , training_size , YAKL_LAMBDA (int ibatch) {
      real x = static_cast<double>(ibatch)/(training_size-1);
      training_inputs (ibatch) = x;
      training_outputs(ibatch) = tanh((x-0.5)*10);
    });
    int batch_size = training_size;

    // Create a model with no ensembles to get the total number of ensembles we'll need
    int num_ensembles = 1;
    auto test = create_inference_model( Matvec<real>( real3d("",1,10,num_ensembles) ) ,
                                        Bias  <real>( real2d("",10,num_ensembles)   ) ,
                                        Relu  <real>( 10 , num_ensembles , 0.1      ) ,
                                        Matvec<real>( real3d("",10,1,num_ensembles) ) ,
                                        Bias  <real>( real2d("",1,num_ensembles)    ) );
    test.init( batch_size , 1 );

    // Create the trainer
    auto num_parameters = test.get_num_trainable_parameters();
    auto num_inputs = 1;
    ponni::Trainer_GD_Nadam_FD<real>  trainer( num_parameters , num_inputs );


    // Initialize the model with the correct number of batches and ensembles
    num_ensembles = trainer.get_num_ensembles();

    // Create model with ensembles
    auto model = create_inference_model( Matvec<real>( real3d("",1,10,num_ensembles) ) ,
                                         Bias  <real>( real2d("",10,num_ensembles)   ) ,
                                         Relu  <real>( 10 , num_ensembles , 0.1      ) ,
                                         Matvec<real>( real3d("",10,1,num_ensembles) ) ,
                                         Bias  <real>( real2d("",1,num_ensembles)    ) );
    model.init( batch_size , num_ensembles );

    // Create model inputs with ensemble dimension added
    real2d model_input("model_input",batch_size,num_ensembles);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(batch_size,num_ensembles) , 
                                      YAKL_LAMBDA (int ibatch, int iens) {
      model_input(ibatch,iens) = training_inputs(ibatch);
    });

    // Train the model
    int num_iter = 100000;
    for (int iter = 0; iter < num_iter; iter++) {
      auto ensemble = trainer.get_ensemble();
      model.set_trainable_parameters( ensemble.get_parameters() );
      auto in = model_input.reshape(1,batch_size,num_ensembles);
      auto model_output = model.forward_batch_parallel( in ).reshape(batch_size,num_ensembles);
      auto loss = ensemble.get_loss();
      parallel_for( YAKL_AUTO_LABEL() , num_ensembles , YAKL_LAMBDA (int iens) {
        real l = 0;
        for (int ibatch = 0; ibatch < batch_size; ibatch++) {
          real adiff = abs( model_output(ibatch,iens) - training_outputs(ibatch) );
          l += adiff*adiff;
        }
        loss(iens) = l / batch_size;
      });
      trainer.update_from_ensemble( ensemble );
      {
        test.set_trainable_parameters( trainer.get_parameters().reshape(num_parameters,1) );
        auto test_output = test.forward_batch_parallel( training_inputs.reshape(1,batch_size,1) ).reshape(batch_size);
        using yakl::componentwise::operator-;
        using yakl::intrinsics::abs;
        using yakl::intrinsics::sum;
        std::cout << "L1: " << sum(abs( training_outputs - test_output )) / sum(abs( training_outputs )) << std::endl;
        if (iter == num_iter-1) std::cout << test_output;
      }
    }

    if (! trainer.parameters_identical_across_tasks()) yakl::yakl_throw("ERROR: parameters are not the same");
  }
  yakl::finalize();
  MPI_Finalize();
}


