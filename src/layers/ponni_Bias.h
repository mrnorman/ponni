
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
      real1d weights;
      bool   trainable;
    };

    Params params;

    Bias () = default;
    ~Bias() = default;
    template < class INIT = Initializer_Random_Uniform<real> >
    Bias( int num_inputs , bool trainable = true , INIT initializer = Initializer_Random_Uniform<real>() ) {
      real1d weights("Bias_weights",num_inputs);
      initializer.fill( weights );
      init(weights,trainable);
    }
    Bias( real1d const &weights , bool trainable=true ) { init(weights,trainable); }


    void init( real1d const &weights , bool trainable=true ) {
      if ( ! weights.initialized() ) Kokkos::abort("ERROR: Bias weights vector not allocated");
      params.weights   = weights;
      params.trainable = trainable;
    }


    char const * get_label() const { return "Bias"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs   (Params const &params_in) { return params_in.weights.extent(0); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs  (Params const &params_in) { return params_in.weights.extent(0); }
    int get_num_inputs               () const { return params.weights.extent(0); }
    int get_num_outputs              () const { return params.weights.extent(0); }
    int get_num_trainable_parameters () const { return params.trainable ? params.weights.extent(0) : 0; }
    int get_array_representation_size() const { return params.weights.size() + 2; }


    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const &input, real2d const &output,
                                                           int ibatch, Params const &params_in) {
      int num_outputs = get_num_outputs(params_in);
      auto &weights = params_in.weights;
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch) + weights(irow);
      }
    }


    void set_trainable_parameters(real1d const &in) {
      if (params.trainable) in.deep_copy_to(params.weights);
    }


    real1d get_trainable_parameters() const {
      if (params.trainable) return params.weights;
      return real1d();
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Bias_weights",get_array_representation_size());
      data(0) = get_num_inputs   ();
      data(1) = params.trainable ? 1 : 0;
      auto weights = params.weights.createHostCopy().collapse();
      for (int i=0; i < weights.size(); i++) { data(2+i) = weights(i); }
      return data;
    }


    void from_array(doubleHost1d const & data) {
      int  num_inputs    = data(0);
      bool trainable     = data(1) == 1;
      realHost1d weights("Bias_weights",num_inputs);
      for (int i=0; i < num_inputs; i++) { weights(i) = data(2+i); }
      init( weights.createDeviceCopy() , trainable );
    }


    void validate() const {
      if (! params.weights.initialized()) Kokkos::abort("ERROR: weights not initialized");
    }

  };

}


