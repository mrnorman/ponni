
#pragma once

#include <algorithm>
#include <random>

namespace ponni {


  template <class T = float>
  class Batch {
    public:
    typedef typename yakl::Array<T  ,1,yakl::memDevice,yakl::styleC> real1d;
    typedef typename yakl::Array<T  ,2,yakl::memDevice,yakl::styleC> real2d;
    typedef typename yakl::Array<int,1,yakl::memDevice,yakl::styleC> int1d;

    Batch () = default;
    ~Batch() = default;
    Batch            (Batch const & rhs) { copy(rhs);               };
    Batch            (Batch const &&rhs) { copy(rhs);               };
    Batch & operator=(Batch const & rhs) { copy(rhs); return *this; };
    Batch & operator=(Batch const &&rhs) { copy(rhs); return *this; };

    Batch(real2d const &a, int1d const &b, real1d const &c) { parameters = a;  global_indices = b;  loss = c; }

    real2d get_parameters    () const { return parameters; }
    real1d get_loss          () const { return loss; }
    int1d  get_global_indices() const { return global_indices; }
    int    get_num_parameters() const { return parameters.extent(0); }
    int    get_batch_size    () const { return parameters.extent(1); }

    protected:
    real2d parameters;
    int1d  global_indices;
    real1d loss;
    void copy(Batch const &rhs) {
      parameters     = rhs.parameters;
      global_indices = rhs.global_indices;
      loss           = rhs.loss;
    }
  };



  template <class T = float , typename std::enable_if<std::is_floating_point<T>::value,bool>::type = true >
  class Trainer_Particle_Swarm {
    public:
    typedef typename yakl::Array<T   ,1,yakl::memDevice,yakl::styleC> real1d;
    typedef typename yakl::Array<T   ,2,yakl::memDevice,yakl::styleC> real2d;
    typedef typename yakl::Array<int ,1,yakl::memDevice,yakl::styleC> int1d;
    typedef typename yakl::Array<T   ,1,yakl::memHost  ,yakl::styleC> realHost1d;
    typedef typename yakl::Array<int ,1,yakl::memHost  ,yakl::styleC> intHost1d;

    Trainer_Particle_Swarm()  = default;
    ~Trainer_Particle_Swarm() = default;
    Trainer_Particle_Swarm            (Trainer_Particle_Swarm const & rhs) { copy(rhs);               };
    Trainer_Particle_Swarm            (Trainer_Particle_Swarm const &&rhs) { copy(rhs);               };
    Trainer_Particle_Swarm & operator=(Trainer_Particle_Swarm const & rhs) { copy(rhs); return *this; };
    Trainer_Particle_Swarm & operator=(Trainer_Particle_Swarm const &&rhs) { copy(rhs); return *this; };


    Trainer_Particle_Swarm( int    num_parameters                  ,
                            int    num_particles        = 1024     ,
                            real1d lbounds_in           = real1d() ,
                            real1d ubounds_in           = real1d() ,
                            T      inertia_in           = 0.8      ,
                            T      velmag_prop_in       = 0.05     ,
                            T      accel_loc_in         = 0.5      ,
                            size_t reset_every_in       = 100      ,
                            T      reset_prop_in        = 0.3      ,
                            size_t rand_seed_counter_in = 0        ) {
      batch_beginning   = 0;
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
      best_loss_per_particle   = std::numeric_limits<T>::max();
      best_loss_overall        = std::numeric_limits<T>::max();
      best_params_per_particle = real2d("best_params_per_particle",num_parameters,num_particles);
      best_params_overall      = real1d("best_params_per_particle",num_parameters              );
      // best_params_per_particle and best_params_overall will be initialized later before use
    }


    void init_particles() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters               , this->parameters               );
      YAKL_SCOPE( velocities               , this->velocities               );
      YAKL_SCOPE( lbounds                  , this->lbounds                  );
      YAKL_SCOPE( ubounds                  , this->ubounds                  );
      YAKL_SCOPE( velmag_prop              , this->velmag_prop              );
      int num_parameters = get_num_parameters();
      int num_particles  = get_num_particles ();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,num_particles) ,
                                        YAKL_LAMBDA (int iparam, int ibatch) {
        yakl::Random prng(iparam*num_particles + ibatch);
        auto param = prng.genFP<T>(lbounds(iparam),ubounds(iparam));
        auto velmag = (ubounds(iparam)-lbounds(iparam))*velmag_prop;
        auto vel   = prng.genFP<T>(-velmag,velmag);
        parameters(iparam,ibatch) = param;
        velocities(iparam,ibatch) = vel  ;
      });
    }


    bool   is_initialized              () const { return parameters.initialized(); }
    int    get_num_parameters          () const { return parameters.extent(0); }
    int    get_num_particles           () const { return parameters.extent(1); }
    real2d get_parameters              () const { return parameters              ; }
    real2d get_velocities              () const { return velocities              ; }
    real2d get_best_params_per_particle() const { return best_params_per_particle; }
    real1d get_best_params_overall     () const { return best_params_overall     ; }
    real1d get_best_loss_per_particle  () const { return best_loss_per_particle  ; }
    T      get_best_loss_overall       () const { return best_loss_overall       ; }
    real1d get_lbounds                 () const { return lbounds                 ; }
    real1d get_ubounds                 () const { return ubounds                 ; }
    size_t get_batch_beginning         () const { return batch_beginning         ; }
    size_t get_num_updates             () const { return num_updates             ; }
    size_t get_rand_seed_counter       () const { return rand_seed_counter       ; }
    T      get_inertia                 () const { return inertia                 ; }
    T      get_accel_loc               () const { return accel_loc               ; }
    T      get_accel_glob              () const { return accel_glob              ; }
    size_t get_reset_every             () const { return reset_every             ; }
    T      get_reset_prop              () const { return reset_prop              ; }
    T      get_velmag_prop             () const { return velmag_prop             ; }
    

    Batch<T> get_batch(int batch_size_in = -1) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_particles  = get_num_particles ();
      auto num_parameters = get_num_parameters();
      int batch_size = batch_size_in > 0 ? batch_size_in : num_particles;

      if (batch_size < 0 || batch_size > num_particles) {
        std::cerr << "ERROR: Calling get_batch with invalid batch_size = [" << batch_size << "]";
        yakl::yakl_throw("");
      }

      real2d params_batch  ("parameters_batch",num_parameters,batch_size);
      int1d  global_indices("global_indices"                 ,batch_size);
      real1d loss_batch    ("loss_batch"                     ,batch_size);

      YAKL_SCOPE( batch_beginning , this->batch_beginning );
      YAKL_SCOPE( parameters      , this->parameters      );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,batch_size) ,
                                        YAKL_LAMBDA (int iparam, int ibatch) {
        int ibatch_glob = batch_beginning + ibatch;
        if (ibatch_glob >= num_particles) ibatch_glob -= num_particles;
        params_batch(iparam,ibatch) = parameters(iparam,ibatch_glob);
        if (iparam == 0) {
          loss_batch    (ibatch) = 0;
          global_indices(ibatch) = ibatch_glob;
        }
      });
      batch_beginning += batch_size;
      if (batch_beginning >= num_particles) batch_beginning -= num_particles;
      return Batch<T>( params_batch , global_indices , loss_batch );
    }


    void update_from_batch( Batch<T> const &batch , MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_particles  = get_num_particles();
      auto num_parameters = get_num_parameters();
      auto batch_size     = batch.get_batch_size();
      auto global_indices = batch.get_global_indices();
      auto batch_losses   = batch.get_loss();
      auto batch_params   = batch.get_parameters();

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

      auto losses_host = batch_losses.createHostCopy();

      /////////////////////////////////////////////////////
      // Update global best loss and parameters
      /////////////////////////////////////////////////////
      auto min_batch_loss = yakl::intrinsics::minval( losses_host ); // Deterministic
      if (min_batch_loss < best_loss_overall) {
        best_loss_overall = min_batch_loss;
        auto min_batch_ind = yakl::intrinsics::minloc( losses_host ); // Deterministic on the host
        parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
          best_params_overall(iparam) = batch_params(iparam,min_batch_ind);
        });
      }
      /////////////////////////////////////////////////////
      // Update per-partical best loss and parameters
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , batch_size , YAKL_LAMBDA (int ibatch) {
        int ibatch_glob = global_indices(ibatch);
        if ( batch_losses(ibatch) < best_loss_per_particle(ibatch_glob) ) {
          best_loss_per_particle(ibatch_glob) = batch_losses(ibatch);
          // This serializes the parameter dimension to reduce the number of required kernel launches
          for (int iparam=0; iparam < num_parameters; iparam++) {
            best_params_per_particle(iparam,ibatch_glob) = batch_params(iparam,ibatch);
          }
        }
      });

      /////////////////////////////////////////////////////
      // Update particle parameter velocities
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,batch_size) ,
                                        YAKL_LAMBDA (int iparam, int ibatch) {
        int ibatch_glob = global_indices(ibatch);
        yakl::Random prng(rand_seed_counter + iparam*batch_size + ibatch);
        auto term_loc  = accel_loc *(1-inertia)*prng.genFP<T>();
        auto term_glob = accel_glob*(1-inertia)*prng.genFP<T>();
        real vel = velocities(iparam,ibatch_glob);
        vel = inertia   * vel                                                                            +
              term_loc  * ( best_params_per_particle(iparam,ibatch_glob) - batch_params(iparam,ibatch) ) +
              term_glob * ( best_params_overall     (iparam            ) - batch_params(iparam,ibatch) );
        velocities(iparam,ibatch_glob) = vel;
        // velocities(iparam,ibatch_glob) = std::min( static_cast<T>(0.5) , std::max( static_cast<T>(-0.5) , vel ) );
      });
      rand_seed_counter += num_parameters*batch_size;

      /////////////////////////////////////////////////////
      // Update particle parameters via velocities
      /////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,batch_size) ,
                                        YAKL_LAMBDA (int iparam, int ibatch) {
        int ibatch_glob = global_indices(ibatch);
        parameters(iparam,ibatch_glob) += velocities(iparam,ibatch_glob);
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
                                          YAKL_LAMBDA (int iparam, int ibatch) {
          int ibatch_glob = sorted_indices(ibatch);
          yakl::Random prng(rand_seed_counter + iparam*num_reset + ibatch);
          auto param = prng.genFP<T>(lbounds(iparam),ubounds(iparam));
          auto velmag = (ubounds(iparam)-lbounds(iparam))*velmag_prop;
          auto vel   = prng.genFP<T>(-velmag,velmag);
          parameters(iparam,ibatch_glob) = param;
          velocities(iparam,ibatch_glob) = vel  ;
          // If we don't reset the best loss, we'll track toward the old particle's best loss params
          best_loss_per_particle(ibatch_glob) = std::numeric_limits<T>::max();
        });
        rand_seed_counter += num_parameters*num_reset;
      }

      num_updates++;
    }

    protected:
    real2d parameters              ;  // parameters being trained                (num_parameters,num_states)
    real2d velocities              ;  // velocity for each particle              (num_parameters,num_states)
    real2d best_params_per_particle;  // best parameters for each particle       (num_parameters,num_states)
    real1d best_params_overall     ;  // best parameters overall                 (num_parameters)
    real1d best_loss_per_particle  ;  // Best loss for each particle             (num_states)
    T      best_loss_overall       ;  // Best loss overall
    real1d lbounds                 ;  // Lower bounds for spawning new particles (num_parameters)
    real1d ubounds                 ;  // Upper bounds for spawning new particles (num_parameters)
    size_t batch_beginning         ;  // The number of batches grabbed so far
    size_t num_updates             ;  // The number of batches grabbed so far
    size_t rand_seed_counter       ;  // Keep MPI tasks in sync by updating rand_seed_counter for random seeds
    T      inertia                 ;  // Degree to which a velocity keeps its old value
    T      accel_loc               ;  // Proportion of non-inertia accel performed toward the particle's best state
    T      accel_glob              ;  // Proportion of non-inertia accel performed toward the global best state
    size_t reset_every             ;  // Reset the lowest "reset_prop" proportion every "reset_every" batches
    T      reset_prop              ;  // Reset the lowest "reset_prop" proportion every "reset_every" batches
    T      velmag_prop             ;  // Proportion of range for each param to use for bounds of random vel init
    void copy(Trainer_Particle_Swarm const &rhs) {
      parameters               = rhs.parameters              ;
      velocities               = rhs.velocities              ;
      best_params_per_particle = rhs.best_params_per_particle;
      best_params_overall      = rhs.best_params_overall     ;
      best_loss_per_particle   = rhs.best_loss_per_particle  ;
      best_loss_overall        = rhs.best_loss_overall       ;
      lbounds                  = rhs.lbounds                 ;
      ubounds                  = rhs.ubounds                 ;
      batch_beginning          = rhs.batch_beginning         ;
      num_updates              = rhs.num_updates             ;
      inertia                  = rhs.inertia                 ;
      accel_loc                = rhs.accel_loc               ;
      accel_glob               = rhs.accel_glob              ;
      reset_every              = rhs.reset_every             ;
      reset_prop               = rhs.reset_prop              ;
      velmag_prop              = rhs.velmag_prop             ;
    }
  };

}


