
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Relu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real max_value;
      real negative_slope;
      real threshold;
      bool trainable;
    };

    Params params;

    Relu() {};
    Relu( int num_inputs , real negative_slope = 0                           ,
                           real threshold = 0                                ,
                           real max_value = std::numeric_limits<real>::max() ,
                           bool trainable = false                            ) {
      init( num_inputs , negative_slope , threshold , max_value , trainable );
    }


    void init( int num_inputs , real negative_slope = 0                           ,
                                real threshold = 0                                ,
                                real max_value = std::numeric_limits<real>::max() ,
                                bool trainable = false                            ) {
      params.num_inputs     = num_inputs    ;
      params.num_outputs    = num_inputs    ;
      params.max_value      = max_value     ;
      params.negative_slope = negative_slope;
      params.threshold      = threshold     ;
      params.trainable      = trainable     ;
    }


    char const * get_label() const { return "Relu"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }
    int get_num_trainable_parameters() const { return params.trainable ? 3 : 0; }
    int get_array_representation_size() const { return 5; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int num_outputs = get_num_outputs(params_in);
      for (int irow = 0; irow < num_outputs; irow++) {
        real max_value      = params_in.max_value     ;
        real negative_slope = params_in.negative_slope;
        real threshold      = params_in.threshold     ;
        real x              = input(irow,ibatch,iens);
        real f_x;
        if      (x >= max_value) { f_x = max_value;                        }
        else if (x >= threshold) { f_x = x;                                }
        else                     { f_x = negative_slope * (x - threshold); }
        output(irow,ibatch,iens) = f_x;
      }
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Relu_params",5);
      data(0) = params.num_inputs;
      data(1) = params.negative_slope;
      data(2) = params.threshold;
      data(3) = params.max_value;
      data(4) = params.trainable ? 1 : 0;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      init( static_cast<int>(data(0)) , static_cast<real>(data(1)) , static_cast<real>(data(2)) , static_cast<real>(data(3)) , data(4) == 1 );
    }


    void validate() const { }

  };

}


