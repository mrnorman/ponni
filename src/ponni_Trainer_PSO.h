
#pragma once

#include <algorithm>
#include <random>

namespace ponni {


  // This represents an ensemble of parameters for the user to get parameters and calculate losses from them
  // The user is not expected to create an instance of this class themselves. Rather, the Trainer class
  // will provide this to the user already created.
  template <class real = float>
  class Ensemble {
    public:
    typedef typename yakl::Array<real,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<int ,1,yakl::memDevice> int1d;

    Ensemble () = default;
    ~Ensemble() = default;
    Ensemble            (Ensemble const & rhs) { copy(rhs);               };
    Ensemble            (Ensemble const &&rhs) { copy(rhs);               };
    Ensemble & operator=(Ensemble const & rhs) { copy(rhs); return *this; };
    Ensemble & operator=(Ensemble const &&rhs) { copy(rhs); return *this; };

    Ensemble(real2d const &a, int1d const &b, real1d const &c) { parameters = a;  global_indices = b;  loss = c; }

    real2d get_parameters    () const { return parameters; }
    real1d get_loss          () const { return loss; }
    int1d  get_global_indices() const { return global_indices; }
    int    get_num_parameters() const { return parameters.extent(0); }
    int    get_ensemble_size () const { return parameters.extent(1); }

    protected:
    real2d parameters;      // Parameters for each particle, dimensioned as num_parameters,num_particles(aka ensemble size)
    int1d  global_indices;  // Global index among all particles for each particle in this ensemble
    real1d loss;            // Loss for each particl ein this ensemble
    void copy(Ensemble const &rhs) {
      parameters     = rhs.parameters;
      global_indices = rhs.global_indices;
      loss           = rhs.loss;
    }
  };



  template <class real = float , typename std::enable_if<std::is_floating_point<real>::value,bool>::type = true >
  class Trainer_Particle_Swarm {
    public:
    typedef typename yakl::Array<real,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<int ,1,yakl::memDevice> int1d;
    typedef typename yakl::Array<real,1,yakl::memHost  > realHost1d;

    Trainer_Particle_Swarm()  = default;
    ~Trainer_Particle_Swarm() = default;
    Trainer_Particle_Swarm            (Trainer_Particle_Swarm const & rhs) { copy(rhs);               };
    Trainer_Particle_Swarm            (Trainer_Particle_Swarm const &&rhs) { copy(rhs);               };
    Trainer_Particle_Swarm & operator=(Trainer_Particle_Swarm const & rhs) { copy(rhs); return *this; };
    Trainer_Particle_Swarm & operator=(Trainer_Particle_Swarm const &&rhs) { copy(rhs); return *this; };


    // Create a Particle Swarm Trainer
    Trainer_Particle_Swarm( int    num_parameters                  ,
                            int    num_particles        = 1024     ,
                            real1d lbounds_in           = real1d() ,    // Lower bound for each parameters
                            real1d ubounds_in           = real1d() ,    // Upper bound for each parameters
                            real   inertia_in           = 0.8      ,    // Velocity inertia
                            real   velmag_prop_in       = 0.05     ,    // Initial velocity magnitude as proportion of ubound-lbound
                            real   accel_loc_in         = 0.5      ,    // Proportion of update controlled by local exploitation
                            size_t reset_every_in       = 100      ,    // N: Reset a proportion of particles every N iterations
                            real   reset_prop_in        = 0.3      ,    // The proportion of particles to reset every N iterations
                            size_t rand_seed_counter_in = 0        ) {  // Random seed for user control
      ensemble_beginning   = 0;
      num_updates       = 0;
      rand_seed_counter = rand_seed_counter_in;
      reset_every       = reset_every_in;
      reset_prop        = reset_prop_in;
      accel_loc         = accel_loc_in;
      accel_glob        = 1 - accel_loc_in;
      inertia           = inertia_in;
      lbounds           = real1d("lbounds",num_parameters);
      ubounds           = real1d("ubounds",num_parameters);
      velmag_prop       = velmag_prop_in;
      if ( lbounds_in.initialized() ) { lbounds = lbounds_in; }
      else                            { lbounds = -2;         }
      if ( ubounds_in.initialized() ) { ubounds = ubounds_in; }
      else                            { ubounds =  2;         }
      parameters = real2d("parameters",num_parameters,num_particles);
      velocities = real2d("velocities",num_parameters,num_particles);
      init_particles();
      rand_seed_counter += num_parameters*num_particles;
      best_loss_per_particle   = real1d("best_loss_per_particle",num_particles);
      best_loss_per_particle   = std::numeric_limits<real>::max();
      best_loss_overall        = std::numeric_limits<real>::max();
      best_params_per_particle = real2d("best_params_per_particle",num_parameters,num_particles);
      best_params_overall      = real1d("best_params_per_particle",num_parameters              );
      // best_params_per_particle and best_params_overall will be initialized later before use
    }



    void init_particles() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters  , this->parameters  );
      YAKL_SCOPE( velocities  , this->velocities  );
      YAKL_SCOPE( lbounds     , this->lbounds     );
      YAKL_SCOPE( ubounds     , this->ubounds     );
      YAKL_SCOPE( velmag_prop , this->velmag_prop );
      int num_parameters = get_num_parameters();
      int num_particles  = get_num_particles ();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,num_particles) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        yakl::Random prng(iparam*num_particles + iens);
        auto param  = prng.genFP<real>(lbounds(iparam),ubounds(iparam));
        auto velmag = (ubounds(iparam)-lbounds(iparam))*velmag_prop;
        auto vel    = prng.genFP<real>(-velmag,velmag);
        parameters(iparam,iens) = param;
        velocities(iparam,iens) = vel  ;
      });
    }



    bool   is_initialized              () const { return parameters.initialized(); }
    int    get_num_parameters          () const { return parameters.extent(0)    ; }
    int    get_num_particles           () const { return parameters.extent(1)    ; }
    real2d get_parameters              () const { return parameters              ; }
    real2d get_velocities              () const { return velocities              ; }
    real2d get_best_params_per_particle() const { return best_params_per_particle; }
    real1d get_best_params_overall     () const { return best_params_overall     ; }
    real1d get_best_loss_per_particle  () const { return best_loss_per_particle  ; }
    real   get_best_loss_overall       () const { return best_loss_overall       ; }
    real1d get_lbounds                 () const { return lbounds                 ; }
    real1d get_ubounds                 () const { return ubounds                 ; }
    size_t get_ensemble_beginning      () const { return ensemble_beginning      ; }
    size_t get_num_updates             () const { return num_updates             ; }
    size_t get_rand_seed_counter       () const { return rand_seed_counter       ; }
    real   get_inertia                 () const { return inertia                 ; }
    real   get_accel_loc               () const { return accel_loc               ; }
    real   get_accel_glob              () const { return accel_glob              ; }
    size_t get_reset_every             () const { return reset_every             ; }
    real   get_reset_prop              () const { return reset_prop              ; }
    real   get_velmag_prop             () const { return velmag_prop             ; }
    


    // Send the user the requested ensemble size of particles. If it's less than the total number of particles,
    // iterations will roll evenly through the particles in contiguous chunks.
    Ensemble<real> get_ensemble(int ensemble_size_in = -1) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_particles  = get_num_particles ();
      auto num_parameters = get_num_parameters();
      int ensemble_size = ensemble_size_in > 0 ? ensemble_size_in : num_particles;

      if (ensemble_size < 0 || ensemble_size > num_particles) {
        std::cerr << "ERROR: Calling get_ensemble with invalid ensemble_size = [" << ensemble_size << "]";
        yakl::yakl_throw("");
      }

      real2d params_ensemble("parameters_ensemble",num_parameters,ensemble_size);
      int1d  global_indices ("global_indices"                    ,ensemble_size);
      real1d loss_ensemble  ("loss_ensemble"                     ,ensemble_size);

      YAKL_SCOPE( ensemble_beginning , this->ensemble_beginning );
      YAKL_SCOPE( parameters         , this->parameters         );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,ensemble_size) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        int iens_glob = ensemble_beginning + iens;
        if (iens_glob >= num_particles) iens_glob -= num_particles;
        params_ensemble(iparam,iens) = parameters(iparam,iens_glob);
        if (iparam == 0) {
          loss_ensemble (iens) = 0;
          global_indices(iens) = iens_glob;
        }
      });
      ensemble_beginning += ensemble_size;
      if (ensemble_beginning >= num_particles) ensemble_beginning -= num_particles;
      return Ensemble<real>( params_ensemble , global_indices , loss_ensemble );
    }



    // Update the particles in this ensemble based on user-provided losses
    // This is guaranteed to give identical updates for each particle for all MPI tasks so long as the provided
    //   losses are identical for all MPI tasks
    void update_from_ensemble( Ensemble<real> const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_particles   = get_num_particles();
      auto num_parameters  = get_num_parameters();
      auto ensemble_size   = ensemble.get_ensemble_size();
      auto global_indices  = ensemble.get_global_indices();
      auto ensemble_losses = ensemble.get_loss();
      auto ensemble_params = ensemble.get_parameters();

      YAKL_SCOPE( parameters               , this->parameters               );
      YAKL_SCOPE( velocities               , this->velocities               );
      YAKL_SCOPE( best_params_per_particle , this->best_params_per_particle );
      YAKL_SCOPE( best_params_overall      , this->best_params_overall      );
      YAKL_SCOPE( best_loss_per_particle   , this->best_loss_per_particle   );
      YAKL_SCOPE( best_loss_overall        , this->best_loss_overall        );
      YAKL_SCOPE( lbounds                  , this->lbounds                  );
      YAKL_SCOPE( ubounds                  , this->ubounds                  );
      YAKL_SCOPE( num_updates              , this->num_updates              );
      YAKL_SCOPE( rand_seed_counter        , this->rand_seed_counter        );
      YAKL_SCOPE( inertia                  , this->inertia                  );
      YAKL_SCOPE( accel_loc                , this->accel_loc                );
      YAKL_SCOPE( accel_glob               , this->accel_glob               );
      YAKL_SCOPE( reset_every              , this->reset_every              );
      YAKL_SCOPE( reset_prop               , this->reset_prop               );
      YAKL_SCOPE( velmag_prop              , this->velmag_prop              );

      // IMPORTANT: Updates must be deterministic to ensure all tasks are using the exact same parameters
      //              without using excess MPI data transfers.

      auto losses_host = ensemble_losses.createHostCopy();

      /////////////////////////////////////////////////////
      // Update global best loss and parameters
      /////////////////////////////////////////////////////
      auto min_ensemble_loss = yakl::intrinsics::minval( losses_host ); // Deterministic
      if (min_ensemble_loss < best_loss_overall) {
        best_loss_overall = min_ensemble_loss;
        auto min_ensemble_ind = yakl::intrinsics::minloc( losses_host ); // Deterministic on the host
        parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
          best_params_overall(iparam) = ensemble_params(iparam,min_ensemble_ind);
        });
      }
      /////////////////////////////////////////////////////
      // Update per-partical best loss and parameters
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , ensemble_size , YAKL_LAMBDA (int iens) {
        int iens_glob = global_indices(iens);
        if ( ensemble_losses(iens) < best_loss_per_particle(iens_glob) ) {
          best_loss_per_particle(iens_glob) = ensemble_losses(iens);
          // This serializes the parameter dimension to reduce the number of required kernel launches
          for (int iparam=0; iparam < num_parameters; iparam++) {
            best_params_per_particle(iparam,iens_glob) = ensemble_params(iparam,iens);
          }
        }
      });

      /////////////////////////////////////////////////////
      // Update particle parameter velocities
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,ensemble_size) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        int iens_glob = global_indices(iens);
        yakl::Random prng(rand_seed_counter + iparam*ensemble_size + iens);
        auto term_loc  = accel_loc *(1-inertia)*prng.genFP<real>();
        auto term_glob = accel_glob*(1-inertia)*prng.genFP<real>();
        real vel = velocities(iparam,iens_glob);
        vel = inertia   * vel                                                                           +
              term_loc  * ( best_params_per_particle(iparam,iens_glob) - ensemble_params(iparam,iens) ) +
              term_glob * ( best_params_overall     (iparam          ) - ensemble_params(iparam,iens) );
        velocities(iparam,iens_glob) = vel;
        // velocities(iparam,iens_glob) = std::min( static_cast<real>(0.5) , std::max( static_cast<real>(-0.5) , vel ) );
      });
      rand_seed_counter += num_parameters*ensemble_size;

      /////////////////////////////////////////////////////
      // Update particle parameters via velocities
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,ensemble_size) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        int iens_glob = global_indices(iens);
        parameters(iparam,iens_glob) += velocities(iparam,iens_glob);
      });

      /////////////////////////////////////////////////////
      // If it's time, kill off the worst and respawn
      // Called infrequently, so optimization is lower
      /////////////////////////////////////////////////////
      if (reset_prop > 0 && num_updates%reset_every == 0) {
        int rank;
        MPI_Comm_rank( comm , &rank   );
        // Sort the particle indices in order of highest loss to lowest
        // Unfortunately, sort() is *not* guaranteed to be deterministic in the C++ stl. Therefore, I need to do
        //   this on the primary task and broadcast to the others.
        int num_reset = (int) (num_particles*reset_prop);
        realHost1d sorted_indices_host("sorted_indices",num_particles);
        if (rank == 0) {
          auto best_loss_per_particle_host = best_loss_per_particle.createHostCopy();
          auto num_particles = get_num_particles();
          for (int i=0; i < num_particles; i++) { sorted_indices_host(i) = i; }
          auto func = [&] (int i, int j) { return best_loss_per_particle_host(i) > best_loss_per_particle_host(j); };
          std::sort( sorted_indices_host.begin() , sorted_indices_host.end() , func );
        }
        MPI_Bcast( sorted_indices_host.data() , num_reset , MPI_INT , 0 , comm );
        // IMPORTANT: After this broadcast, only the first num_reset values are defined. Using subset_slowest_dimension
        auto sorted_indices = sorted_indices_host.subset_slowest_dimension(num_reset).createDeviceCopy();

        // Reset the worst particles to random values to increase exploration
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,num_reset) ,
                                          YAKL_LAMBDA (int iparam, int iens) {
          int iens_glob = sorted_indices(iens);
          yakl::Random prng(rand_seed_counter + iparam*num_reset + iens);
          auto param = prng.genFP<real>(lbounds(iparam),ubounds(iparam));
          auto velmag = (ubounds(iparam)-lbounds(iparam))*velmag_prop;
          auto vel   = prng.genFP<real>(-velmag,velmag);
          parameters(iparam,iens_glob) = param;
          velocities(iparam,iens_glob) = vel  ;
          // If we don't reset the best loss, we'll track toward the old particle's best loss params
          best_loss_per_particle(iens_glob) = std::numeric_limits<real>::max();
        });
        rand_seed_counter += num_parameters*num_reset;
      }

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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(get_num_parameters(),get_num_particles()) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        if ( parameters(iparam,iens) != parameters_min(iparam,iens) ||
             parameters(iparam,iens) != parameters_max(iparam,iens)  ) is_same = false;
      });
      return is_same.hostRead();
    }



    protected:

    real2d parameters              ;  // parameters being trained                (num_parameters,num_states)
    real2d velocities              ;  // velocity for each particle              (num_parameters,num_states)
    real2d best_params_per_particle;  // best parameters for each particle       (num_parameters,num_states)
    real1d best_params_overall     ;  // best parameters overall                 (num_parameters)
    real1d best_loss_per_particle  ;  // Best loss for each particle             (num_states)
    real   best_loss_overall       ;  // Best loss overall
    real1d lbounds                 ;  // Lower bounds for spawning new particles (num_parameters)
    real1d ubounds                 ;  // Upper bounds for spawning new particles (num_parameters)
    size_t ensemble_beginning         ;  // The number of ensemblees grabbed so far
    size_t num_updates             ;  // The number of ensemblees grabbed so far
    size_t rand_seed_counter       ;  // Keep MPI tasks in sync by updating rand_seed_counter for random seeds
    real   inertia                 ;  // Degree to which a velocity keeps its old value
    real   accel_loc               ;  // Proportion of non-inertia accel performed toward the particle's best state
    real   accel_glob              ;  // Proportion of non-inertia accel performed toward the global best state
    size_t reset_every             ;  // Reset the lowest "reset_prop" proportion every "reset_every" ensemblees
    real   reset_prop              ;  // Reset the lowest "reset_prop" proportion every "reset_every" ensemblees
    real   velmag_prop             ;  // Proportion of range for each param to use for bounds of random vel init

    void copy(Trainer_Particle_Swarm const &rhs) {
      parameters               = rhs.parameters              ;
      velocities               = rhs.velocities              ;
      best_params_per_particle = rhs.best_params_per_particle;
      best_params_overall      = rhs.best_params_overall     ;
      best_loss_per_particle   = rhs.best_loss_per_particle  ;
      best_loss_overall        = rhs.best_loss_overall       ;
      lbounds                  = rhs.lbounds                 ;
      ubounds                  = rhs.ubounds                 ;
      ensemble_beginning       = rhs.ensemble_beginning      ;
      num_updates              = rhs.num_updates             ;
      rand_seed_counter        = rhs.rand_seed_counter       ;
      inertia                  = rhs.inertia                 ;
      accel_loc                = rhs.accel_loc               ;
      accel_glob               = rhs.accel_glob              ;
      reset_every              = rhs.reset_every             ;
      reset_prop               = rhs.reset_prop              ;
      velmag_prop              = rhs.velmag_prop             ;
    }
  };

}


