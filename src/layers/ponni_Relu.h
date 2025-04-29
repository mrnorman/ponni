
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float>
  struct Relu {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    real static constexpr huge = std::numeric_limits<real>::max();

    struct Params {
      int  num_inputs     ;
      real negative_slope ;
    };

    Params params;

    Relu() {};
    Relu( int num_inputs , real negative_slope=0 ) {
      init(num_inputs,negative_slope);
    }


    void init( int num_inputs , real negative_slope=0 ) {
      params.num_inputs     = num_inputs    ;
      params.negative_slope = negative_slope;
    }


    char const * get_label() const { return "Relu"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs   (Params const &params_in) { return params_in.num_inputs; }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs  (Params const &params_in) { return params_in.num_inputs; }
    int get_num_inputs   () const { return params.num_inputs; }
    int get_num_outputs  () const { return params.num_inputs; }
    int get_num_trainable_parameters() const { return 0; }
    int get_array_representation_size() const { return 2; }


    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const &input, real2d const &output,
                                                           int ibatch, Params const &params_in) {
      int  num_outputs = get_num_outputs(params_in);
      real negative_slope = params_in.negative_slope;
      for (int irow = 0; irow < num_outputs; irow++) {
        real f_x = input(irow,ibatch);
        if (f_x < 0) f_x *= negative_slope;
        output(irow,ibatch) = f_x;
      }
    }


    void set_trainable_parameters(real2d const &in) { }


    real1d get_trainable_parameters() const { return real1d(); }


    doubleHost1d to_array() const {
      doubleHost1d data("Relu_params",get_array_representation_size());
      data(0) = params.num_inputs;
      data(1) = params.negative_slope;
      return data;
    }


    void from_array(doubleHost1d const &data) { init(data(0),data(1)); }


    void validate() const { }

  };

}


