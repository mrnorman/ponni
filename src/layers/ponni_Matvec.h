
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Matvec {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      real3d weights;
      bool   trainable;
      real   lb;
      real   ub;
    };

    Params params;

    Matvec () = default;
    ~Matvec() = default;
    template < class INIT = Initializer_Random_Uniform<real> >
    Matvec( int num_inputs , int num_outputs , int num_ensembles = 1 , bool trainable = true , real lb = -2 , real ub = 2 ,
            INIT initializer = Initializer_Random_Uniform<real>() ) {
      real3d weights("Bias_weights",num_inputs,num_outputs,num_ensembles);
      initializer.fill( weights );
      init(weights,trainable,lb,ub);
    }
    Matvec( real2d const &weights , bool trainable=true , real lb=-2, real ub=2 ) { init(weights,trainable,lb,ub); }
    Matvec( real3d const &weights , bool trainable=true , real lb=-2, real ub=2 ) { init(weights,trainable,lb,ub); }


    void init( real2d const &weights , bool trainable=true , real lb=-2, real ub=2 ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.weights   = weights.reshape(weights.extent(0),weights.extent(1),1);
      params.trainable = trainable;
      params.lb        = lb;
      params.ub        = ub;
    }
    void init( real3d const &weights , bool trainable=true , real lb=-2, real ub=2 ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.weights   = weights;
      params.trainable = trainable;
      params.lb        = lb;
      params.ub        = ub;
    }


    char const * get_label() const { return "Matvec"; }
    YAKL_INLINE static int get_num_inputs   (Params const &params_in) { return params_in.weights.extent(0); }
    YAKL_INLINE static int get_num_outputs  (Params const &params_in) { return params_in.weights.extent(1); }
    YAKL_INLINE static int get_num_ensembles(Params const &params_in) { return params_in.weights.extent(2); }
    real1d get_lbounds() const { real1d ret("Matvec_lb",get_num_trainable_parameters());  ret = params.lb;  return ret; }
    real1d get_ubounds() const { real1d ret("Matvec_ub",get_num_trainable_parameters());  ret = params.ub;  return ret; }
    int    get_num_inputs               () const { return params.weights.extent(0); }
    int    get_num_outputs              () const { return params.weights.extent(1); }
    int    get_num_ensembles            () const { return params.weights.extent(2); }
    int    get_num_trainable_parameters () const { return params.trainable ? params.weights.extent(0)*
                                                                             params.weights.extent(1)  : 0; }
    int    get_array_representation_size() const { return params.weights.size() + 6; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int num_inputs  = get_num_inputs (params_in);
      int num_outputs = get_num_outputs(params_in);
      auto &weights = params_in.weights;
      for (int irow = 0; irow < num_outputs; irow++) {
        real tmp = 0;
        for (int k=0; k < num_inputs; k++) { tmp += weights(k,irow,iens) * input(k,ibatch,iens); }
        output(irow,ibatch,iens) = tmp;
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
      if (params.trainable) return params.weights.reshape(get_num_inputs()*get_num_outputs(),get_num_ensembles());
      return real2d();
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Matvec_weights",get_array_representation_size());
      data(0) = get_num_inputs   ();
      data(1) = get_num_outputs  ();
      data(2) = get_num_ensembles();
      data(3) = params.trainable ? 1 : 0;
      data(4) = params.lb;
      data(5) = params.ub;
      auto weights = params.weights.createHostCopy().collapse();
      for (int i=0; i < weights.size(); i++) { data(6+i) = weights(i); }
      return data;
    }


    void from_array(doubleHost1d const & data) {
      int  num_inputs    = data(0);
      int  num_outputs   = data(1);
      int  num_ensembles = data(2);
      bool trainable     = data(3) == 1;
      real lb            = data(4);
      real ub            = data(5);
      realHost1d weights("Matvec_weights",num_inputs*num_outputs*num_ensembles);
      for (int i=0; i < weights.size(); i++) { weights(i) = data(6+i); }
      init( weights.createDeviceCopy().reshape(num_inputs,num_outputs,num_ensembles) , trainable , lb , ub );
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


