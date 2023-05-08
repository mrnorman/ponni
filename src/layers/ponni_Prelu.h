
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Prelu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real alpha;
      real threshold;
      bool trainable;
    };

    Params params;

    Prelu() {};
    Prelu( int num_inputs , real alpha=1.67326 , real threshold=0 , bool trainable = false ) {
      init( num_inputs , alpha , threshold , trainable );
    }


    void init( int num_inputs , real alpha=1.67326 , real threshold=0 , bool trainable = false ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.alpha       = alpha;
      params.threshold   = threshold;
      params.trainable   = trainable;
    }


    char const * get_label         () const { return "Prelu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int get_num_trainable_parameters() const { return params.trainable ? 2 : 0; }
    int get_array_representation_size() const { return 4; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        real alpha     = params_in.alpha;
        real threshold = params_in.threshold;
        real x         = input(irow,ibatch,iens);
        if (x < threshold) { output(irow,ibatch,iens) = alpha*x; }
        else               { output(irow,ibatch,iens) = x;       }
      }
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Prelu_params",4);
      data(0) = params.num_inputs;
      data(1) = params.alpha;
      data(2) = params.threshold;
      data(3) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) , static_cast<real>(data(1)) , static_cast<real>(data(2)) , data(3) == 1 );
    }


    void validate() const { }

  };

}


