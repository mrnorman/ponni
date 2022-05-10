
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


    void init( real2d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec weights matrix not initialized");
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[1];
      params.weights     = weights;
    }


    char const * get_label      () const { return "Matvec"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      auto &weights = params.weights;
      real tmp = 0;
      for (int k=0; k < params.num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
      output(irow,ibatch) = tmp;
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

  };



  class Matvec_trainable {
  public:
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int              num_inputs;
      int              num_outputs;
      real2d_trainable weights;
    };

    Params params;

    Matvec_trainable() {}
    Matvec_trainable(realHost2d const &weights) { init(weights); }


    void init( realHost2d const &weights ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Matvec_trainable weights matrix not initialized");
      params.num_inputs  = weights.dimension[0];
      params.num_outputs = weights.dimension[1];
      params.weights     = real2d_trainable("weights",params.num_inputs,params.num_outputs);
      for (int i=0; i < weights.totElems(); i++) { params.weights.data()[i] = weights.data()[i]; }
    }


    char const * get_label() const { return "Matvec_trainable"; }
    int get_num_inputs () const { return params.num_inputs ; }
    int get_num_outputs() const { return params.num_outputs; }


    static void compute_one_output(Params const &params, realConst2d_trainable input, real2d_trainable const &output,
                                   int ibatch, int irow) {
      auto &weights = params.weights;
      real_trainable tmp = 0;
      for (int k=0; k < params.num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
      output(irow,ibatch) = tmp;
    }


    void print_verbose() const {
      std::cout << "    weights:\n";
      for (int irow=0; irow < params.num_outputs; irow++) {
        std::cout << "      ";
        for (int icol=0; icol < params.num_inputs; icol++) {
          std::cout << std::setw(12) << std::setprecision(4) << params.weights(icol,irow) << "  ";
        }
        std::cout << "\n";
      }
    }

  };

}


