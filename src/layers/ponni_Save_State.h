
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N, class real = float>
  struct Save_State {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = true;
    int  static constexpr index           = N;

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Save_State () = default;
    ~Save_State() = default;
    Save_State( int num_inputs ) { init( num_inputs ); }


    void init( int num_inputs ) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    char const * get_label() const { return "Save_State"; }
    YAKL_INLINE static int get_num_inputs   (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs  (Params const &params_in) { return params_in.num_outputs; }
    YAKL_INLINE static int get_num_ensembles(Params const &params_in) { return 1; }
    real1d get_lbounds                  () const { return real1d(); }
    real1d get_ubounds                  () const { return real1d(); }
    int    get_num_inputs               () const { return params.num_inputs ; }
    int    get_num_outputs              () const { return params.num_outputs; }
    int    get_num_ensembles            () const { return 1; }
    int    get_num_trainable_parameters () const { return 0; }
    int    get_array_representation_size() const { return 2; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch,iens) = input(irow,ibatch,iens);
      }
    }


    void set_trainable_parameters(real2d const &in, bool fence = true) { if (fence) yakl::fence(); }


    doubleHost1d to_array() const {
      doubleHost1d data("Save_State_params",get_array_representation_size());
      data(0) = get_num_inputs();
      data(1) = N;
      return data;
    }


    void from_array(doubleHost1d const &data) {
      if (data(1) != N) yakl::yakl_throw("ERROR: Save_State saved state index incompatible with data from file");
      init( static_cast<int>(data(0)) );
    }


    void validate() const { }

  };

}


