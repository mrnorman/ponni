
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Matvec {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      real3d weights;
      bool   trainable;
    };

    Params params;

    Matvec()  = default;
    ~Matvec() = default;
    Matvec( real2d const &weights , bool trainable = true ) { init( weights , trainable ); }
    Matvec( real3d const &weights , bool trainable = true ) { init( weights , trainable ); }


    void init( real2d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.weights   = weights.reshape(weights.extent(0),weights.extent(1),1);
      params.trainable = trainable;
    }
    void init( real3d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.weights   = weights;
      params.trainable = trainable;
    }


    char const * get_label() const { return "Matvec"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.weights.extent(0); }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.weights.extent(1); }
    int get_num_trainable_parameters() const { return params.trainable ? params.weights.extent(0)*params.weights.extent(1) : 0; }
    int get_array_representation_size() const { return params.weights.size() + 4; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int num_inputs  = get_num_inputs (params_in);
      int num_outputs = get_num_outputs(params_in);
      for (int irow = 0; irow < num_outputs; irow++) {
        auto &weights = params_in.weights;
        real tmp = 0;
        for (int k=0; k < num_inputs; k++) { tmp += weights(k,irow,iens) * input(k,ibatch,iens); }
        output(irow,ibatch,iens) = tmp;
      }
    }


    doubleHost1d to_array() const {
      auto weights_host = params.weights.createHostCopy().collapse();
      doubleHost1d data("Matvec_weights",get_array_representation_size());
      data(0) = params.weights.extent(0);
      data(1) = params.weights.extent(1);
      data(2) = params.weights.extent(2);
      data(3) = params.trainable ? 1 : 0;
      for (int i=0; i < weights_host.size(); i++) { data(4+i) = weights_host(i); }
      return data;
    }


    void from_array(doubleHost1d const & data) {
      realHost1d weights_host("Matvec_weights",data(0)*data(1)*data(2));
      for (int i=0; i < weights_host.size(); i++) { weights_host(i) = data(4+i); }
      auto weights = weights_host.createDeviceCopy().reshape(static_cast<int>(data(0)),
                                                             static_cast<int>(data(1)),
                                                             static_cast<int>(data(2)));
      init(weights,data(3) == 1);
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


