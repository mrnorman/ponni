
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Selu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real alpha;
      real lambda;
      real threshold;
      bool trainable;
    };

    Params params;

    Selu() {};
    Selu( int num_inputs , real alpha=1.67326 , real lambda=1.0507 , real threshold=0 , bool trainable = false) {
      init( num_inputs , alpha , lambda , threshold , trainable );
    }


    void init( int num_inputs , real alpha=1.67326 , real lambda=1.0507 , real threshold=0 , bool trainable = false ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.lambda      = lambda;
      params.threshold   = threshold;
      params.trainable   = trainable;
    }


    char const * get_label         () const { return "Selu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        real alpha     = params_in.alpha;
        real lambda    = params_in.lambda;
        real threshold = params_in.threshold;
        real x         = input(irow,ibatch);
        if (x < threshold) { output(irow,ibatch) = lambda * alpha * ( std::exp(x) - 1 ); }
        else               { output(irow,ibatch) = lambda * x; }
      }
    }


    void print_verbose() const {
      std::cout << "    alpha:     " << params.alpha     << "\n";
      std::cout << "    lambda:    " << params.lambda    << "\n";
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    int get_num_trainable_parameters() const { return params.trainable ? 3 : 0; }


    int get_array_representation_size() const { return 5; }


    doubleHost1d to_array() const {
      doubleHost1d data("Relu_params",5);
      data(0) = params.num_inputs;
      data(1) = params.alpha;
      data(2) = params.lambda;
      data(3) = params.threshold;
      data(4) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) , static_cast<real>(data(1)) , static_cast<real>(data(2)) , static_cast<real>(data(3)) , data(4) == 1 );
    }


    void validate() const { }

  };

}


