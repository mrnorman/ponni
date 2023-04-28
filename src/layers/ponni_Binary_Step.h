
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Binary_Step {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real threshold;
      bool trainable;
    };

    Params params;

    Binary_Step() {};
    Binary_Step( int num_inputs , real threshold = 0 , bool trainable = false ) {
      init( num_inputs , threshold , trainable );
    }


    void init( int num_inputs , real threshold = 0 , bool trainable = false ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
      params.threshold   = threshold ;
      params.trainable   = trainable ;
    }


    char const * get_label         () const { return "Binary Step"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch) >= params_in.threshold ? 1 : 0;
      }
    }

    void print_verbose() const {
      std::cout << "    threshold: " << params.threshold << "\n";
    }


    int get_num_trainable_parameters() const { return params.trainable ? 1 : 0; }


    doubleHost1d to_array() const {
      doubleHost1d data("Binary_Step_params",3);
      data(0) = params.num_inputs;
      data(1) = params.threshold;
      data(2) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) , static_cast<real>(data(1)) , data(2) == 1 );
    }


    void validate() const { }

  };

}


