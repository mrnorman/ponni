
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Bias {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      real2d weights;
      bool   trainable;
      real   lb;
      real   ub;
    };

    Params params;

    Bias () = default;
    ~Bias() = default;
    template < class INIT = Initializer_Random_Uniform<real> >
    Bias( int num_inputs , int num_ensembles = 1 , INIT initializer = Initializer_Random_Uniform<real>() ,
          bool trainable = true , real lb = -2 , real ub = 2 ) {
      real2d weights("Bias_weights",num_inputs,num_ensembles);
      initializer.fill( weights );
      init(weights,trainable,lb,ub);
    }
    Bias( real1d const &weights , bool trainable=true , real lb=-2 , real ub=2 ) { init(weights,trainable,lb,ub); }
    Bias( real2d const &weights , bool trainable=true , real lb=-2 , real ub=2 ) { init(weights,trainable,lb,ub); }


    void init( real1d const &weights , bool trainable=true , real lb=-2, real ub=2 ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights   = weights.reshape(weights.extent(0),1);
      params.trainable = trainable;
      params.lb        = lb;
      params.ub        = ub;
    }
    void init( real2d const &weights , bool trainable=true , real lb=-2, real ub=2 ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights   = weights;
      params.trainable = trainable;
      params.lb        = lb;
      params.ub        = ub;
    }


    char const * get_label() const { return "Bias"; }
    YAKL_INLINE static int get_num_inputs   (Params const &params_in) { return params_in.weights.extent(0); }
    YAKL_INLINE static int get_num_outputs  (Params const &params_in) { return params_in.weights.extent(0); }
    YAKL_INLINE static int get_num_ensembles(Params const &params_in) { return params_in.weights.extent(1); }
    real1d get_lbounds() const { real1d ret("Bias_lb",get_num_trainable_parameters());  ret = params.lb;  return ret; }
    real1d get_ubounds() const { real1d ret("Bias_ub",get_num_trainable_parameters());  ret = params.ub;  return ret; }
    int get_num_inputs   () const { return params.weights.extent(0); }
    int get_num_outputs  () const { return params.weights.extent(0); }
    int get_num_ensembles() const { return params.weights.extent(1); }
    int get_num_trainable_parameters() const { return params.trainable ? params.weights.extent(0) : 0; }
    int get_array_representation_size() const { return params.weights.size() + 5; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int num_outputs = get_num_outputs(params_in);
      auto &weights = params_in.weights;
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch,iens) = input(irow,ibatch,iens) + weights(irow,iens);
      }
    }


    void set_trainable_parameters(real2d const &in, bool fence = true) {
      if (params.trainable) {
        auto tmp = in.collapse();
        tmp.subset_slowest_dimension(params.weights.size()).deep_copy_to(params.weights);
      }
      if (fence) yakl::fence();
    }


    real2d get_trainable_parameters() const {
      if (params.trainable) return params.weights;
      return real2d();
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Bias_weights",get_array_representation_size());
      data(0) = get_num_inputs   ();
      data(1) = get_num_ensembles();
      data(2) = params.trainable ? 1 : 0;
      data(3) = params.lb;
      data(4) = params.ub;
      auto weights = params.weights.createHostCopy().collapse();
      for (int i=0; i < weights.size(); i++) { data(5+i) = weights(i); }
      return data;
    }


    void from_array(doubleHost1d const & data) {
      int  num_inputs    = data(0);
      int  num_ensembles = data(1);
      bool trainable     = data(2) == 1;
      real lb            = data(3);
      real ub            = data(4);
      realHost1d weights("Bias_weights",num_inputs*num_ensembles);
      for (int i=0; i < num_inputs*num_ensembles; i++) { weights(i) = data(5+i); }
      init( weights.createDeviceCopy().reshape(num_inputs,num_ensembles) , trainable , lb , ub );
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


