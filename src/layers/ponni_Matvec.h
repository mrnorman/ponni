
#pragma once
// Included by ponni.h

namespace ponni {

  template <class real = float, int N_IN = 1, int N_OUT = 1>
  struct Matvec {
    typedef yakl::Array<double * ,Kokkos::HostSpace> doubleHost1d;
    typedef yakl::Array<real   * ,Kokkos::HostSpace> realHost1d;
    typedef yakl::Array<real   *                   > real1d;
    typedef yakl::Array<real   **                  > real2d;
    
    bool static constexpr overwrite_input = false;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    int static constexpr INPUT_SIZE  = static_cast<int>(N_IN );
    int static constexpr OUTPUT_SIZE = static_cast<int>(N_OUT);

    struct Params {
      real2d weights;
      bool   trainable;
    };

    Params params;

    Matvec () = default;
    ~Matvec() = default;
    template < class INIT = Initializer_Random_Uniform<real> >
    Matvec( int num_inputs , int num_outputs , bool trainable = true ,
            INIT initializer = Initializer_Random_Uniform<real>() ) {
      real2d weights("Bias_weights",num_inputs,num_outputs);
      initializer.fill( weights );
      init(weights,trainable);
    }
    Matvec( real2d const &weights , bool trainable=true ) { init(weights,trainable); }

    void init( real2d const &weights , bool trainable=true ) {
      if ( ! weights.is_allocated() ) Kokkos::abort("ERROR: Matvec weights matrix not is_allocated");
      params.weights   = weights;
      params.trainable = trainable;
    }

    char const * get_label() const { return "Matvec"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs (Params const &params_in) { return params_in.weights.extent(0); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const &params_in) { return params_in.weights.extent(1); }
    int    get_num_inputs               () const { return params.weights.extent(0); }
    int    get_num_outputs              () const { return params.weights.extent(1); }
    int    get_num_trainable_parameters () const { return params.trainable ? params.weights.size() : 0; }
    int    get_array_representation_size() const { return params.weights.size() + 3; }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( real2d const & input     ,
                                                            real2d const & output    ,
                                                            int            ibatch    ,
                                                            Params const & params_in ) {
      int num_inputs  = get_num_inputs (params_in);
      int num_outputs = get_num_outputs(params_in);
      auto &weights = params_in.weights;
      for (int irow = 0; irow < num_outputs; irow++) {
        real tmp = 0;
        for (int k=0; k < num_inputs; k++) { tmp += weights(k,irow) * input(k,ibatch); }
        output(irow,ibatch) = tmp;
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs( SArray<real,N_IN > const & input     ,
                                                            SArray<real,N_OUT>       & output    ,
                                                            Params             const & params_in ) {
      for (int irow = 0; irow < N_OUT; irow++) {
        real tmp = 0;
        for (int k=0; k < N_IN; k++) { tmp += params_in.weights(k,irow) * input(k); }
        output(irow) = tmp;
      }
    }

    void set_trainable_parameters(real1d const &in) {
      if (params.trainable) in.subset_slowest_dimension(get_num_trainable_parameters()).deep_copy_to(params.weights.collapse());
    }

    real1d get_trainable_parameters() const {
      if (params.trainable) return params.weights.reshape(get_num_inputs()*get_num_outputs());
      return real1d();
    }

    doubleHost1d to_array() const {
      doubleHost1d data("Matvec_weights",get_array_representation_size());
      data(0) = get_num_inputs   ();
      data(1) = get_num_outputs  ();
      data(2) = params.trainable ? 1 : 0;
      auto weights = params.weights.createHostCopy().collapse();
      for (int i=0; i < weights.size(); i++) { data(3+i) = weights(i); }
      return data;
    }

    void from_array(doubleHost1d const & data) {
      int  num_inputs    = data(0);
      int  num_outputs   = data(1);
      bool trainable     = data(2) == 1;
      realHost1d weights_flat("Matvec_weights",num_inputs*num_outputs);
      for (int i=0; i < weights_flat.size(); i++) { weights_flat(i) = data(3+i); }
      real2d weights("Matvec_weights",num_inputs,num_outputs);
      weights_flat.createDeviceCopy().deep_copy_to(weights.collapse());
      init( weights , trainable );
    }

    void validate() const {
      if (! params.weights.is_allocated()) Kokkos::abort("ERROR: weights not is_allocated");
    }
  };

}


