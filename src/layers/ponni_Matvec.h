
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Matvec {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real2d weights;
      bool   trainable;
    };

    Params params;

    Matvec()  = default;
    ~Matvec() = default;
    Matvec( real2d const &weights , bool trainable = true ) {
      init( weights , trainable );
    }


    void init( real2d const &weights , bool trainable = true ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.num_inputs  = weights.extent(0);
      params.num_outputs = weights.extent(1);
      params.weights     = weights;
      params.trainable   = trainable;
    }


    char const * get_label         () const { return "Matvec"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        auto &weights = params_in.weights;
        real tmp = 0;
        for (int k=0; k < params_in.num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
        output(irow,ibatch) = tmp;
      }
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      auto kernel_host = params.weights.createHostCopy();
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      ";
        for (int icol=0; icol < params.num_inputs; icol++) {
          std::cout << std::setw(12) << std::setprecision(4) << kernel_host(icol,irow) << "  ";
        }
        std::cout << "\n";
      }
    }


    int get_num_trainable_parameters() const { return params.trainable ? params.weights.size() : 0; }


    doubleHost1d to_array() const {
      auto weights_host = params.weights.createHostCopy().collapse();
      doubleHost1d data("Matvec_weights",weights_host.size() + 3);
      for (int i=0; i < weights_host.size(); i++) { data(i) = weights_host(i); }
      data(weights_host.size()  ) = params.weights.extent(0);
      data(weights_host.size()+1) = params.weights.extent(1);
      data(weights_host.size()+2) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const & data) {
      realHost1d weights_host("Matvec_weights",data.size()-3);
      for (int i=0; i < weights_host.size(); i++) { weights_host(i) = data(i); }
      auto weights = weights_host.createDeviceCopy().reshape(static_cast<int>(data(data.size()-3)),
                                                             static_cast<int>(data(data.size()-2)));
      init(weights,data(data.size()-1) == 1);
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


