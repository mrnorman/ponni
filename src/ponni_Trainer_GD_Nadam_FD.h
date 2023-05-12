
#pragma once

#include <algorithm>
#include <random>
#include <map>

namespace ponni {

  template <class real = float>
  class Trainer_GD_Nadam_FD {
    public:
    typedef typename yakl::Array<real,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<int ,1,yakl::memDevice> int1d;
    real static constexpr tiny = std::numeric_limits<real>::min()*10;
    real static constexpr eps  = std::numeric_limits<real>::epsilon();

    class Ensemble {
      public:
      Ensemble () = default;
      ~Ensemble() = default;
      Ensemble            (Ensemble const & rhs) { copy(rhs);               };
      Ensemble            (Ensemble const &&rhs) { copy(rhs);               };
      Ensemble & operator=(Ensemble const & rhs) { copy(rhs); return *this; };
      Ensemble & operator=(Ensemble const &&rhs) { copy(rhs); return *this; };

      Ensemble(real2d const &a, real1d const &b) { parameters = a;  loss = b; }

      real2d get_parameters    () const { return parameters; }
      real1d get_loss          () const { return loss; }
      int    get_num_parameters() const { return parameters.extent(0); }
      int    get_ensemble_size () const { return parameters.extent(1); }

      protected:
      real2d parameters; // Parameters for each particle, dimensioned as num_parameters,ensemble_size
      real1d loss;       // Loss for each particl ein this ensemble
      void copy(Ensemble const &rhs) {
        parameters = rhs.parameters;
        loss       = rhs.loss;
      }
    };

    Trainer_GD_Nadam_FD()  = default;
    ~Trainer_GD_Nadam_FD() = default;
    Trainer_GD_Nadam_FD            (Trainer_GD_Nadam_FD const & rhs) { copy(rhs);               };
    Trainer_GD_Nadam_FD            (Trainer_GD_Nadam_FD const &&rhs) { copy(rhs);               };
    Trainer_GD_Nadam_FD & operator=(Trainer_GD_Nadam_FD const & rhs) { copy(rhs); return *this; };
    Trainer_GD_Nadam_FD & operator=(Trainer_GD_Nadam_FD const &&rhs) { copy(rhs); return *this; };


    // Create a Particle Swarm Trainer
    Trainer_GD_Nadam_FD( int  num_parameters  ,
                         int  num_inputs = -1 ,
                         real alpha = 0.001   ,
                         real mu    = 0.9     ,
                         real nu    = 0.999   ) {
      this->num_updates = 0;
      this->alpha       = alpha;
      this->mu          = mu;
      this->nu          = nu;
      this->m           = real1d("m",num_parameters);
      this->n           = real1d("n",num_parameters);
      this->m           = 0;
      this->n           = 0;
      this->parameters  = real1d("parameters",num_parameters);
      std::random_device          rd {};
      std::mt19937                gen {rd()};
      std::normal_distribution<>  d;
      if (num_inputs > 0) { d = std::normal_distribution<>{0,std::sqrt(2./num_inputs)}; } // He initialization
      else                { d = std::normal_distribution<>{0,0.1}; }                      // He for num_inputs == 100
      auto parameters_host = parameters.createHostCopy();
      for (int i=0; i < parameters.size(); i++) { parameters_host(i) = d(gen); }
      this->parameters = parameters_host.createDeviceCopy();
    }



    bool   is_initialized    () const { return parameters.initialized(); }
    int    get_num_parameters() const { return parameters.extent(0)    ; }
    int    get_num_ensembles () const { return parameters.extent(0)+1  ; }
    real1d get_parameters    () const { return parameters              ; }
    size_t get_num_updates   () const { return num_updates             ; }
    real   get_alpha         () const { return alpha                   ; }
    real   get_mu            () const { return mu                      ; }
    real   get_nu            () const { return nu                      ; }
    


    // Gives the user an ensemble of parameters to calculate loss functions with
    Ensemble get_ensemble() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters , this->parameters );
      auto num_parameters = get_num_parameters();
      // Calculate 2-norm of the data
      real1d norm2_vals("norm2_vals",num_parameters);
      parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
        norm2_vals(iparam) = parameters(iparam)*parameters(iparam);
      });
      // Calculate delta for finite-differences
      real norm2 = std::sqrt( yakl::intrinsics::sum(norm2_vals) );
      real delta = std::pow( eps/2 , 1./3.) / (norm2 + 1.e-7);
      // Create parameter ensemble
      int ensemble_size = num_parameters+1;
      real2d ensemble_parameters("ensemble_parameters",num_parameters,ensemble_size);
      real1d ensemble_loss      ("ensemble_loss"                     ,ensemble_size);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,ensemble_size) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        ensemble_parameters(iparam,iens) = parameters(iparam);
        if (iparam == iens) ensemble_parameters(iparam,iens) += delta;
        if (iparam == 0) ensemble_loss(iens) = 0;
      });
      return Ensemble( ensemble_parameters , ensemble_loss );
    }



    // Update the particles in this ensemble based on user-provided losses
    // This is guaranteed to give identical updates for each particle for all MPI tasks so long as the provided
    //   losses are identical for all MPI tasks
    void update_from_ensemble( Ensemble const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_parameters      = get_num_parameters();
      auto ensemble_losses     = ensemble.get_loss();
      auto ensemble_parameters = ensemble.get_parameters();

      YAKL_SCOPE( parameters , this->parameters );
      YAKL_SCOPE( alpha      , this->alpha      );
      YAKL_SCOPE( mu         , this->mu         );
      YAKL_SCOPE( nu         , this->nu         );
      YAKL_SCOPE( m          , this->m          );
      YAKL_SCOPE( n          , this->n          );

      parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
        real g = ( ensemble_losses    (       iparam) - ensemble_losses    (       num_parameters) ) /
                 ( ensemble_parameters(iparam,iparam) - ensemble_parameters(iparam,num_parameters) );
        m(iparam) = mu * m(iparam) + (1-mu) * g;
        n(iparam) = nu * n(iparam) + (1-nu) * g*g;
        real mhat = mu*m(iparam)/(1-mu) + g;
        real nhat = nu*n(iparam)/(1-nu);
        parameters(iparam) -= alpha / (sqrt(nhat) + 1.e-7) * mhat;
      });
      num_updates++;
    }



    bool parameters_identical_across_tasks( MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters , this->parameters );
      auto parameters_host     = parameters.createHostCopy();
      auto parameters_min_host = parameters.createHostObject();
      auto parameters_max_host = parameters.createHostObject();
      MPI_Datatype data_type;
      if constexpr (std::is_same<real,float >::value) { data_type = MPI_FLOAT;  }
      if constexpr (std::is_same<real,double>::value) { data_type = MPI_DOUBLE; }
      MPI_Allreduce( parameters_host.data() , parameters_min_host.data() , parameters.size() , data_type , MPI_MIN , comm );
      MPI_Allreduce( parameters_host.data() , parameters_max_host.data() , parameters.size() , data_type , MPI_MAX , comm );
      auto parameters_min = parameters_min_host.createDeviceCopy();
      auto parameters_max = parameters_max_host.createDeviceCopy();
      yakl::ScalarLiveOut<bool> is_same(true);
      parallel_for( YAKL_AUTO_LABEL() , get_num_parameters() , YAKL_LAMBDA (int iparam) {
        if ( parameters(iparam) != parameters_min(iparam) ||
             parameters(iparam) != parameters_max(iparam)  ) is_same = false;
      });
      return is_same.hostRead();
    }



    protected:

    real1d parameters ;  // parameters being trained (num_parameters)
    size_t num_updates;  // The number of model updates performes thus far
    real   alpha      ;
    real   mu         ;
    real   nu         ;
    real1d m          ;
    real1d n          ;

    void copy(Trainer_GD_Nadam_FD const &rhs) {
      parameters  = rhs.parameters       ;
      num_updates = rhs.num_updates      ;
      alpha       = rhs.alpha            ;
      mu          = rhs.mu               ;
      nu          = rhs.nu               ;
      m           = rhs.m                ;
      n           = rhs.n                ;
    }
  };

}


