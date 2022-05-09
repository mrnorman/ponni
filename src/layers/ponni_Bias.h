
#pragma once
// Included by ponni.h

namespace ponni {

  class Bias {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real1d weights;
    };

    Params params;

    Bias() {}
    Bias(real1d const &weights) { init(weights); }
    ~Bias() { params.weights = real1d(); }


    void init( real1d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Bias weights vector not initialized");
      params.weights     = weights;
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[0];
    }


    char const * get_label      () const { return "Bias"; }
    int          get_num_inputs () const { return params.num_inputs ; }
    int          get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      output(irow,ibatch) = input(irow,ibatch) + params.weights(irow);
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      auto bias_host = params.weights.createHostCopy();
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      " << std::setw(12) << bias_host(irow) << "\n";
      }
    }

  };



  class Bias_train {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      yakl::Array<autodiff::Variable<real>,1,yakl::memHost,yakl::styleC> weights;
    };

    Params params;


    void print_verbose() { }

  };

}


