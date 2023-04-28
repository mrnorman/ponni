
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
  yakl::finalize();
  MPI_Finalize();
}

