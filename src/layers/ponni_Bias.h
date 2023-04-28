
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Bias {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real1d weights;
      bool   trainable;
    };

    Params params;

    Bias() {}
    Bias( real1d const &weights , bool trainable = true ) {
      init( weights , trainable );
    }


    void init( real1d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.num_inputs  = weights.extent(0);
      params.num_outputs = weights.extent(0);
      params.weights     = weights;
      params.trainable   = trainable;
    }


    char const * get_label         () const { return "Bias"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch) + params_in.weights(irow);
      }
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      auto bias_host = params.weights.createHostCopy();
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      " << std::setw(12) << bias_host(irow) << "\n";
      }
    }


    int get_num_trainable_parameters() const { return params.trainable ? params.weights.size() : 0; }


    doubleHost1d to_array() const {
      auto weights_host = params.weights.createHostCopy();
      doubleHost1d data("Bias_weights",weights_host.size()+1);
      for (int i=0; i < weights_host.size(); i++) { data(i) = weights_host(i); }
      data(weights_host.size()) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const & data) {
      realHost1d weights_host("Bias_weights",data.size());
      for (int i=0; i < weights_host.size(); i++) { weights_host(i) = data(i); }
      auto weights = weights_host.createDeviceCopy();
      init(weights,data(data.size()-1) == 1);
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


