
#include "ponni.h"
#include "ponni_Trainer.h"

int main( int argc , char **argv ) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    using yakl::c::parallel_for;
    int    num_iterations = 100;
    int    num_particles  = 10;
    int    num_parameters = 2;
    float  inertia        = 0.8;
    float  velmag_prop    = 0.01;
    float  accel_loc      = 0.5;
    size_t reset_every    = 100;
    float  reset_prop     = 0.3;

    yakl::Array<float,1,yakl::memDevice,yakl::styleC> lbounds("lbounds",num_parameters);
    yakl::Array<float,1,yakl::memDevice,yakl::styleC> ubounds("ubounds",num_parameters);
    lbounds = 0;
    ubounds = 5;

    auto func = YAKL_LAMBDA (float x, float y) {
      return (x-3.14)*(x-3.14) + (y-2.72)*(y-2.72) + sin(3*x+1.41) + sin(4*y-1.73);
    };

    ponni::Trainer_Particle_Swarm pso( num_parameters ,
                                       num_particles  ,
                                       lbounds        ,
                                       ubounds        ,
                                       inertia        ,
                                       velmag_prop    ,
                                       accel_loc      ,
                                       reset_every    ,
                                       reset_prop     );

    for (int iter = 0; iter < num_iterations; iter++) {
      auto batch  = pso.get_batch();
      auto params = batch.get_parameters();
      auto loss   = batch.get_loss();

      parallel_for( YAKL_AUTO_LABEL() , batch.get_batch_size() , YAKL_LAMBDA (int ibatch) {
        loss(ibatch) = func(params(0,ibatch),params(1,ibatch));
      });

      pso.update_from_batch( batch );
    }
    std::cout << std::setprecision(10) << pso.get_best_loss_overall() - (-1.808351994) << std::endl;
    if (std::abs(pso.get_best_loss_overall() - (-1.808351994)) > 1.e-6) {
      std::cerr << "ERROR: Did not converge" << std::endl;   yakl::yakl_throw("");
    }

    if (!pso.parameters_identical_across_tasks()) yakl::yakl_throw("ERROR: parameters are not the same");
  }
  {
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;
    typedef float real;
    typedef yakl::Array<real,1,yakl::memDevice> real1d;
    typedef yakl::Array<real,2,yakl::memDevice> real2d;
    typedef yakl::Array<real,3,yakl::memDevice> real3d;
    int batch_size = 1024*16;
    int num_inputs  = 1;
    int num_outputs = 1;
    yakl::Array<real,1> training_inputs ("training_inputs" ,batch_size);
    yakl::Array<real,1> training_outputs("training_outputs",batch_size);
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , batch_size , YAKL_LAMBDA (int i) {
      training_inputs (i) = static_cast<double>(i)/(batch_size-1);
      training_outputs(i) = tanh((training_inputs(i)-0.5)*10);
    });
    int num_ensembles = 4096;
    int neurons = 16;
    real1d dummy("dummy",num_ensembles);
    auto model = ponni::create_inference_model( Matvec<real>( real3d("mat",num_inputs,neurons,num_ensembles) )  ,
                                                Bias  <real>( real2d("mat"           ,neurons,num_ensembles) )  ,
                                                Relu  <real>( neurons , dummy , dummy , dummy , true )          ,
                                                Save_State<0,real>( neurons )                                   ,
                                                Matvec<real>( real3d("mat",neurons,neurons,num_ensembles) )     ,
                                                Bias  <real>( real2d("mat"        ,neurons,num_ensembles) )     ,
                                                Relu  <real>( neurons , dummy , dummy , dummy , true )          ,
                                                Binop_Add <0,real>( neurons )                                   ,
                                                Matvec<real>( real3d("mat",neurons,num_outputs,num_ensembles) ) ,
                                                Bias  <real>( real2d("mat"        ,num_outputs,num_ensembles) ) );
    int    num_iterations = 1000;
    int    num_particles  = num_ensembles;
    int    num_parameters = model.get_num_trainable_parameters();
    float  inertia        = 0.8;
    float  velmag_prop    = 0.01;
    float  accel_loc      = 0.5;
    size_t reset_every    = 100;
    float  reset_prop     = 0.3;
    auto lbounds        = model.get_lbounds();
    auto ubounds        = model.get_ubounds();
    ponni::Trainer_Particle_Swarm pso( num_parameters ,
                                       num_particles  ,
                                       lbounds        ,
                                       ubounds        ,
                                       inertia        ,
                                       velmag_prop    ,
                                       accel_loc      ,
                                       reset_every    ,
                                       reset_prop     );
    for (int iter = 0; iter < num_iterations; iter++) {
      auto batch  = pso.get_batch(num_ensembles);
      auto params = batch.get_parameters();
      auto loss   = batch.get_loss();
      model.set_trainable_parameters(params);
      real3d inputs ("inputs" ,num_inputs,batch_size,num_ensembles);
      real3d outputs("outputs",num_inputs,batch_size,num_ensembles);
      YAKL_SCOPE( model_params , model.params );
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<2>(batch_size,num_ensembles) ,
                                                 YAKL_LAMBDA (int ibatch, int iens) {
        inputs(0,ibatch,iens) = training_inputs(ibatch);
        model.forward_in_kernel( inputs , outputs , model_params , ibatch , iens );
        loss(ibatch) = std::abs( training_outputs(ibatch) - outputs(0,ibatch,iens) );
      });

      pso.update_from_batch( batch );
    }
  }
  yakl::finalize();
  MPI_Finalize();
}


