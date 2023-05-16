
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
    using ponni::Save_State;
    using ponni::Binop_Add;
    typedef float real;
    typedef yakl::Array<real,1,yakl::memDevice> real1d;
    typedef yakl::Array<real,2,yakl::memDevice> real2d;
    typedef yakl::Array<real,3,yakl::memDevice> real3d;

    // Create training data
    int training_size = 1024*1024*10;
    real1d training_inputs ("training_inputs" ,training_size);
    real1d training_outputs("training_outputs",training_size);
    parallel_for( YAKL_AUTO_LABEL() , training_size , YAKL_LAMBDA (int ibatch) {
      real x = static_cast<double>(ibatch)/(training_size-1);
      training_inputs (ibatch) = x;
      training_outputs(ibatch) = tanh((x-0.5)*10);
    });
    int batch_size = 1024;
    int num_batches = training_size / batch_size;

    // Create a model with no ensembles to get the total number of ensembles we'll need
    int  num_inputs          = 1;
    int  num_outputs         = 1;
    int  num_neurons         = 10;
    int  num_ensembles       = 1;
    real relu_negative_slope = 0.1;
    real relu_threshold      = 0;
    real relu_max_value      = std::numeric_limits<real>::max();
    auto test = create_inference_model( Matvec<real>      ( num_inputs,num_neurons,num_ensembles          ) ,
                                        Bias  <real>      ( num_neurons,num_ensembles                     ) ,
                                        Relu  <real>      ( num_neurons,num_ensembles,relu_negative_slope,relu_threshold,relu_max_value,true ) ,
                                        Save_State<0,real>( num_neurons                                   ) ,
                                        Matvec<real>      ( num_neurons,num_neurons,num_ensembles         ) ,
                                        Bias  <real>      ( num_neurons,num_ensembles                     ) ,
                                        Relu  <real>      ( num_neurons,num_ensembles,relu_negative_slope,relu_threshold,relu_max_value,true ) ,
                                        Binop_Add<0,real> ( num_neurons                                   ) ,
                                        Matvec<real>      ( num_neurons,num_outputs,num_ensembles         ) ,
                                        Bias  <real>      ( num_outputs,num_ensembles                     ) );
    test.init( training_size , 1 );

    // Create the trainer
    auto num_parameters = test.get_num_trainable_parameters();
    ponni::Trainer_GD_Adam_FD<real> trainer( test.get_trainable_parameters().reshape(num_parameters) );

    // Initialize the model with the correct number of batches and ensembles
    num_ensembles = trainer.get_num_ensembles();

    // Create model with ensembles
    auto model = create_inference_model( Matvec<real>      ( num_inputs,num_neurons,num_ensembles          ) ,
                                         Bias  <real>      ( num_neurons,num_ensembles                     ) ,
                                         Relu  <real>      ( num_neurons,num_ensembles,relu_negative_slope,relu_threshold,relu_max_value,true ) ,
                                         Save_State<0,real>( num_neurons                                   ) ,
                                         Matvec<real>      ( num_neurons,num_neurons,num_ensembles         ) ,
                                         Bias  <real>      ( num_neurons,num_ensembles                     ) ,
                                         Relu  <real>      ( num_neurons,num_ensembles,relu_negative_slope,relu_threshold,relu_max_value,true ) ,
                                         Binop_Add<0,real> ( num_neurons                                   ) ,
                                         Matvec<real>      ( num_neurons,num_outputs,num_ensembles         ) ,
                                         Bias  <real>      ( num_outputs,num_ensembles                     ) );
    model.init( batch_size , num_ensembles );
    model.print();

    // Create model inputs with ensemble dimension added
    real2d model_input("model_input",batch_size,num_ensembles);

    // Train the model
    int num_epochs = 5;
    yakl::timer_start("training_time");
    for (int iepoch = 0; iepoch < num_epochs; iepoch++) {
      ponni::shuffle_training_data( training_inputs .reshape(1,training_size) ,
                                    training_outputs.reshape(1,training_size) , iepoch+1 );
      for (int batch_id = 0; batch_id < num_batches; batch_id++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(batch_size,num_ensembles) , 
                                          YAKL_LAMBDA (int ibatch, int iens) {
          int ibatch_glob = std::min(training_size-1,batch_id*batch_size + ibatch);
          model_input(ibatch,iens) = training_inputs(ibatch_glob);
        });
        auto ensemble = trainer.get_ensemble();
        model.set_trainable_parameters( ensemble.get_parameters() );
        auto in = model_input.reshape(1,batch_size,num_ensembles);
        auto model_output = model.forward_batch_parallel( in ).reshape(batch_size,num_ensembles);
        auto loss = ensemble.get_loss();
        parallel_for( YAKL_AUTO_LABEL() , num_ensembles , YAKL_LAMBDA (int iens) {
          real l = 0;
          for (int ibatch = 0; ibatch < batch_size; ibatch++) {
            int ibatch_glob = std::min(training_size-1,batch_id*batch_size + ibatch);
            real adiff = abs( model_output(ibatch,iens) - training_outputs(ibatch_glob) );
            l += adiff*adiff;
          }
          loss(iens) = l / batch_size;
        });
        trainer.update_from_ensemble( ensemble );
      } // ibatch
      trainer.increment_epoch();
      // std::cout << "Epoch: " << iepoch << "\n";
    }
    yakl::timer_stop("training_time");
    {
      test.set_trainable_parameters( trainer.get_parameters().reshape(num_parameters,1) );
      auto test_output = test.forward_batch_parallel( training_inputs.reshape(1,training_size,1) ).reshape(training_size);
      using yakl::componentwise::operator-;
      using yakl::intrinsics::abs;
      using yakl::intrinsics::sum;
      std::cout << "MAE: "
                << sum(abs( training_outputs - test_output )) / training_outputs.size() << std::endl;
    }

    if (! trainer.parameters_identical_across_tasks()) yakl::yakl_throw("ERROR: parameters are not the same");
  }
  yakl::finalize();
  MPI_Finalize();
}


