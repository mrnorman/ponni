
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
    };

    Params params;

    Bias() {}
    Bias( real1d const &weights , bool trainable = true ) { init( weights , trainable ); }
    Bias( real2d const &weights , bool trainable = true ) { init( weights , trainable ); }


    void init( real1d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights   = weights.reshape(weights.extent(0),1);
      params.trainable = trainable;
    }
    void init( real2d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights   = weights;
      params.trainable = trainable;
    }


    char const * get_label         () const { return "Bias"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.weights.extent(0); }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.weights.extent(0); }
    int get_num_trainable_parameters() const { return params.trainable ? params.weights.extent(0) : 0; }
    int get_array_representation_size() const { return params.weights.size() + 3; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int num_outputs = get_num_outputs(params_in);
      for (int irow = 0; irow < num_outputs; irow++) {
        output(irow,ibatch,iens) = input(irow,ibatch,iens) + params_in.weights(irow,iens);
      }
    }


    doubleHost1d to_array() const {
      auto weights_host = params.weights.createHostCopy().collapse();
      doubleHost1d data("Bias_weights",get_array_representation_size());
      data(0) = params.weights.extent(0);
      data(1) = params.weights.extent(1);
      data(2) = params.trainable ? 1 : 0;
      for (int i=0; i < weights_host.size(); i++) { data(3+i) = weights_host(i); }
      return data;
    }


    void from_array(doubleHost1d const & data) {
      realHost1d weights_host("Bias_weights",data(0)*data(1));
      for (int i=0; i < weights_host.size(); i++) { weights_host(i) = data(3+i); }
      auto weights = weights_host.createDeviceCopy().reshape(data(0),data(1));
      init(weights,data(2) == 1);
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


