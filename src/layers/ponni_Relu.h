
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
      int    num_inputs     ;
      real1d negative_slope ;
      real1d threshold      ;
      real1d max_value      ;
      bool   trainable      ;
      real lb_negative_slope;
      real ub_negative_slope;
      real lb_threshold     ;
      real ub_threshold     ;
      real lb_max_value     ;
      real ub_max_value     ;
    };

    Params params;

    Relu() {};
    Relu( int num_inputs , real negative_slope=0 , real threshold=0 , real max_value=huge , bool trainable=false ,
          real lb_negative_slope = 0    , real ub_negative_slope = 1    ,
          real lb_threshold      = -2   , real ub_threshold      = 2    ,
          real lb_max_value      = huge , real ub_max_value      = huge ) {
      init(num_inputs,negative_slope,threshold,max_value,trainable,lb_negative_slope,ub_negative_slope,
           lb_threshold,ub_threshold,lb_max_value,ub_max_value);
    }
    Relu( int num_inputs , real1d negative_slope , real1d threshold ,  real1d max_value , bool trainable=false ,
          real lb_negative_slope = 0    , real ub_negative_slope = 1    ,
          real lb_threshold      = -2   , real ub_threshold      = 2    ,
          real lb_max_value      = huge , real ub_max_value      = huge ) {
      init(num_inputs,negative_slope,threshold,max_value,trainable,lb_negative_slope,ub_negative_slope,
           lb_threshold,ub_threshold,lb_max_value,ub_max_value);
    }


    void init( int num_inputs , real negative_slope=0 , real threshold=0 , real max_value=huge , bool trainable=false ,
               real lb_negative_slope = 0    , real ub_negative_slope = 1    ,
               real lb_threshold      = -2   , real ub_threshold      = 2    ,
               real lb_max_value      = huge , real ub_max_value      = huge ) {
      params.num_inputs        = num_inputs                     ;
      params.negative_slope    = real1d("Relu_negative_slope",1);
      params.threshold         = real1d("Relu_threshold"     ,1);
      params.max_value         = real1d("Relu_max_value"     ,1);
      params.negative_slope    = negative_slope                 ;
      params.threshold         = threshold                      ;
      params.max_value         = max_value                      ;
      params.trainable         = trainable                      ;
      params.lb_negative_slope = lb_negative_slope              ;
      params.ub_negative_slope = ub_negative_slope              ;
      params.lb_threshold      = lb_threshold                   ;
      params.ub_threshold      = ub_threshold                   ;
      params.lb_max_value      = lb_max_value                   ;
      params.ub_max_value      = ub_max_value                   ;
    }
    void init( int num_inputs , real1d negative_slope , real1d threshold ,  real1d max_value , bool trainable=false ,
               real lb_negative_slope = 0    , real ub_negative_slope = 1    ,
               real lb_threshold      = -2   , real ub_threshold      = 2    ,
               real lb_max_value      = huge , real ub_max_value      = huge ) {
      params.num_inputs        = num_inputs                     ;
      params.negative_slope    = negative_slope                 ;
      params.threshold         = threshold                      ;
      params.max_value         = max_value                      ;
      params.trainable         = trainable                      ;
      params.lb_negative_slope = lb_negative_slope              ;
      params.ub_negative_slope = ub_negative_slope              ;
      params.lb_threshold      = lb_threshold                   ;
      params.ub_threshold      = ub_threshold                   ;
      params.lb_max_value      = lb_max_value                   ;
      params.ub_max_value      = ub_max_value                   ;
    }


    char const * get_label() const { return "Relu"; }
    YAKL_INLINE static int get_num_inputs   (Params const &params_in) { return params_in.num_inputs; }
    YAKL_INLINE static int get_num_outputs  (Params const &params_in) { return params_in.num_inputs; }
    YAKL_INLINE static int get_num_ensembles(Params const &params_in) { return params_in.negative_slope.extent(0); }
    real1d get_lbounds() const {
      realHost1d lbounds("Bias_lb",get_num_trainable_parameters());
      lbounds(0) = params.lb_negative_slope;
      lbounds(1) = params.lb_threshold     ;
      lbounds(2) = params.lb_max_value     ;
      return lbounds.createDeviceCopy();
    }
    real1d get_ubounds() const {
      realHost1d ubounds("Bias_ub",get_num_trainable_parameters());
      ubounds(0) = params.ub_negative_slope;
      ubounds(1) = params.ub_threshold     ;
      ubounds(2) = params.ub_max_value     ;
      return ubounds.createDeviceCopy();
    }
    int get_num_inputs   () const { return params.num_inputs; }
    int get_num_outputs  () const { return params.num_inputs; }
    int get_num_ensembles() const { return params.negative_slope.extent(0); }
    int get_num_trainable_parameters() const { return params.trainable ? 3 : 0; }
    int get_array_representation_size() const { return 3*get_num_ensembles()+9; }


    YAKL_INLINE static void compute_all_outputs(real3d const &input, real3d const &output,
                                                int ibatch, int iens, Params const &params_in) {
      int  num_outputs    = get_num_outputs(params_in);
      real negative_slope = params_in.negative_slope(iens);
      real threshold      = params_in.threshold     (iens);
      real max_value      = params_in.max_value     (iens);
      for (int irow = 0; irow < num_outputs; irow++) {
        real x              = input(irow,ibatch,iens);
        real f_x;
        if      (x >= max_value) { f_x = max_value;                        }
        else if (x >= threshold) { f_x = x;                                }
        else                     { f_x = negative_slope * (x - threshold); }
        output(irow,ibatch,iens) = f_x;
      }
    }


    void set_trainable_parameters(real2d const &in, bool fence = true) {
      int nens = get_num_ensembles();
      auto tmp = in.collapse();
      tmp.subset_slowest_dimension(0*nens,1*nens-1).deep_copy_to(params.negative_slope);
      tmp.subset_slowest_dimension(1*nens,2*nens-1).deep_copy_to(params.threshold     );
      tmp.subset_slowest_dimension(2*nens,3*nens-1).deep_copy_to(params.max_value     );
      if (fence) yakl::fence();
    }


    doubleHost1d to_array() const {
      doubleHost1d data("Relu_params",get_array_representation_size());
      int nens = get_num_ensembles();
      data(0) = nens;
      data(1) = get_num_inputs();
      data(2) = params.trainable ? 1 : 0;
      data(3) = params.lb_negative_slope;
      data(4) = params.ub_negative_slope;
      data(5) = params.lb_threshold     ;
      data(6) = params.ub_threshold     ;
      data(7) = params.lb_max_value     ;
      data(8) = params.ub_max_value     ;
      auto negative_slope = params.negative_slope.createHostCopy();
      auto threshold      = params.threshold     .createHostCopy();
      auto max_value      = params.max_value     .createHostCopy();
      for (int i=0; i < nens; i++) {
        data(9+0*nens+i) = negative_slope(i);
        data(9+1*nens+i) = threshold     (i);
        data(9+2*nens+i) = max_value     (i);
      }
      return data;
    }


    void from_array(doubleHost1d const &data) {
      int  nens              = data(0);
      int  num_inputs        = data(1);
      bool trainable         = data(2) == 1;
      real lb_negative_slope = data(3);
      real ub_negative_slope = data(4);
      real lb_threshold      = data(5);
      real ub_threshold      = data(6);
      real lb_max_value      = data(7);
      real ub_max_value      = data(8);
      realHost1d negative_slope("Relu_negative_slope",nens);
      realHost1d threshold     ("Relu_threshold     ",nens);
      realHost1d max_value     ("Relu_max_value     ",nens);
      for (int i=0; i < nens; i++) {
        negative_slope(i) = data(9       +i);
        threshold     (i) = data(9+  nens+i);
        max_value     (i) = data(9+2*nens+i);
      }
      init( num_inputs , negative_slope.createDeviceCopy() , threshold.createDeviceCopy() ,
            max_value.createDeviceCopy() , trainable );
    }


    void validate() const { }

  };

}


