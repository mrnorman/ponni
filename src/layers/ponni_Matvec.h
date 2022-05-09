
#pragma once
// Included by ponni.h

namespace ponni {

  class Matvec {
  public:
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      real2d weights;
    };

    Params params;

    Matvec() {}
    Matvec(real2d const &weights) { init(weights); }
    ~Matvec() { params.weights = real2d(); }


    void init( real2d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[1];
      params.weights     = weights;
    }


    char const * get_label      () const { return "Matvec"; }
    int          get_num_inputs () const { return params.num_inputs ; }
    int          get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      auto &weights = params.weights;
      int num_inputs = weights.dimension[0];
      real tmp = 0;
      for (int k=0; k < num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
      output(irow,ibatch) = tmp;
    }


    void print_verbose() { }

  };



  class Matvec_train {
  public:
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int    num_inputs;
      int    num_outputs;
      yakl::Array<autodiff::Variable<real>,2,yakl::memHost,yakl::styleC> weights;
    };

    Params params;


    void print_verbose() { }

  };

}


