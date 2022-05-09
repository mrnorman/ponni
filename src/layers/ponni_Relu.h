
#pragma once
// Included by ponni.h

namespace ponni {

  class Relu {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int  num_inputs;
      int  num_outputs;
      real max_value;
      real negative_slope;
      real threshold;
    };

    Params params;

    Relu() {};
    Relu( int num_inputs , real max_value = std::numeric_limits<real>::max() ,
                           real negative_slope = 0 ,
                           real threshold = 0 ) {
      init(num_inputs, max_value, negative_slope, threshold);
    }


    char const * get_label      () const { return "Relu"; }
    int          get_num_inputs () const { return params.num_inputs ; }
    int          get_num_outputs() const { return params.num_outputs; }


    void init(int num_inputs , real max_value = std::numeric_limits<real>::max() ,
                               real negative_slope = 0 ,
                               real threshold = 0 ) {
      params.num_inputs     = num_inputs    ;
      params.num_outputs    = num_inputs    ;
      params.max_value      = max_value     ;
      params.negative_slope = negative_slope;
      params.threshold      = threshold     ;
    }


    YAKL_INLINE static void compute_one_output(Params const &params, realConst2d input, real2d const &output,
                                               int ibatch, int irow) {
      real max_value      = params.max_value     ;
      real negative_slope = params.negative_slope;
      real threshold      = params.threshold     ;
      real x              = input(irow,ibatch);
      real f_x;
      if      (x >= max_value) { f_x = max_value;                        }
      else if (x >= threshold) { f_x = x;                                }
      else                     { f_x = negative_slope * (x - threshold); }
      output(irow,ibatch) = f_x;
    }


    void print_verbose() const {
      std::cout << "    max_value:      " << params.max_value      << "\n";
      std::cout << "    negative_slope: " << params.negative_slope << "\n";
      std::cout << "    threshold:      " << params.threshold      << "\n";
    }

  };



  class Relu_train {
  public:

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?

    struct Params {
      int                      num_inputs;
      int                      num_outputs;
      autodiff::Variable<real> max_value, negative_slope, threshold;
      bool                     train_max_value, train_negative_slope, train_threshold;
    };

    Params params;



    void print_verbose() { }

  };

}


