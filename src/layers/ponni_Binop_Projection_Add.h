#pragma once
// Included by ponni.h

namespace ponni {

  template <int ISAVE, class real = float, int NIN = 1, int NSAVE = 1>
  struct Binop_Projection_Add {
    typedef Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,Kokkos::HostSpace > realHost1d;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,ponni::DeviceSpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,ponni::DeviceSpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true;
    bool static constexpr save            = false;
    int  static constexpr index           = ISAVE;

    int static constexpr INPUT_SIZE  = static_cast<int>(NIN);
    int static constexpr OUTPUT_SIZE = static_cast<int>(NIN);

    struct Params {
      real2d weights;
      real1d bias;
      bool   trainable;
    };

    Params params;

    Binop_Projection_Add() = default;
    ~Binop_Projection_Add() = default;

    template <class INIT = Initializer_Random_Uniform<real> >
    Binop_Projection_Add(int num_inputs, int num_saved_inputs, bool trainable = true,
                         INIT initializer = Initializer_Random_Uniform<real>()) {
      real2d weights("Projection_skip_weights", num_saved_inputs, num_inputs);
      real1d bias("Projection_skip_bias", num_inputs);
      initializer.fill(weights);
      initializer.fill(bias);
      init(weights, bias, trainable);
    }

    Binop_Projection_Add(real2d const & weights, real1d const & bias, bool trainable = true) {
      init(weights, bias, trainable);
    }

    void init(real2d const & weights, real1d const & bias, bool trainable = true) {
      if (!weights.is_allocated() || !bias.is_allocated()) {
        Kokkos::abort("ERROR: Binop_Projection_Add weights/bias not allocated");
      }
      if (weights.extent(1) != bias.extent(0)) Kokkos::abort("ERROR: Binop_Projection_Add incompatible weights/bias shapes");
      params.weights = weights;
      params.bias = bias;
      params.trainable = trainable;
    }

    char const * get_label() const { return "Binop_Projection_Add"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.weights.extent(1); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.weights.extent(1); }
    int get_num_inputs() const { return params.weights.extent(1); }
    int get_num_outputs() const { return params.weights.extent(1); }
    int get_num_trainable_parameters() const { return params.trainable ? params.weights.size() + params.bias.size() : 0; }
    int get_array_representation_size() const { return 4 + params.weights.size() + params.bias.size(); }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(real2d const & input,
                                                           real2d const & saved,
                                                           real2d const & output,
                                                           int ibatch,
                                                           Params const & params_in) {
      int num_saved = params_in.weights.extent(0);
      int num_outputs = params_in.weights.extent(1);
      for (int i = 0; i < num_outputs; i++) {
        real proj = params_in.bias(i);
        for (int k = 0; k < num_saved; k++) proj += params_in.weights(k,i) * saved(k,ibatch);
        output(i,ibatch) = input(i,ibatch) + proj;
      }
    }

    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<real,NIN> const & input,
                                                           ponni::SArray<real,NSAVE> const & saved,
                                                           ponni::SArray<real,NIN> & output,
                                                           Params const & params_in) {
      int num_saved = params_in.weights.extent(0);
      int num_outputs = params_in.weights.extent(1);
      for (int i = 0; i < num_outputs; i++) {
        real proj = params_in.bias(i);
        for (int k = 0; k < num_saved; k++) proj += params_in.weights(k,i) * saved(k);
        output(i) = input(i) + proj;
      }
    }

    void set_trainable_parameters(real1d const & in) {
      if (params.trainable) {
        int nweights = params.weights.size();
        int nbias = params.bias.size();
        if (in.extent(0) < nweights + nbias) Kokkos::abort("ERROR: Binop_Projection_Add trainable input too small");
        Kokkos::deep_copy(ponni::flatten(params.weights), Kokkos::subview(in, std::pair<int,int>(0, nweights)));
        Kokkos::deep_copy(params.bias, Kokkos::subview(in, std::pair<int,int>(nweights, nweights + nbias)));
      }
    }

    real1d get_trainable_parameters() const {
      if (!params.trainable) return real1d();
      int nweights = params.weights.size();
      int nbias = params.bias.size();
      real1d out("Projection_skip_trainable", nweights + nbias);
      Kokkos::deep_copy(Kokkos::subview(out, std::pair<int,int>(0, nweights)), ponni::flatten(params.weights));
      Kokkos::deep_copy(Kokkos::subview(out, std::pair<int,int>(nweights, nweights + nbias)), params.bias);
      return out;
    }

    doubleHost1d to_array() const {
      int num_saved = params.weights.extent(0);
      int num_outputs = params.weights.extent(1);
      auto weights_h = ponni::create_host_copy(params.weights);
      auto bias_h = ponni::create_host_copy(params.bias);
      auto weights_flat = ponni::flatten(weights_h);
      doubleHost1d data("Projection_skip_array", get_array_representation_size());
      data(0) = num_saved;
      data(1) = num_outputs;
      data(2) = params.trainable ? 1 : 0;
      data(3) = ISAVE;
      for (int i = 0; i < weights_flat.extent(0); i++) data(4 + i) = weights_flat(i);
      for (int i = 0; i < bias_h.extent(0); i++) data(4 + weights_flat.extent(0) + i) = bias_h(i);
      return data;
    }

    void from_array(doubleHost1d const & data) {
      if (data(3) != ISAVE) Kokkos::abort("ERROR: Binop_Projection_Add saved state index incompatible with data from file");
      int num_saved = static_cast<int>(data(0));
      int num_outputs = static_cast<int>(data(1));
      bool trainable = data(2) == 1;
      real2d weights("Projection_skip_weights", num_saved, num_outputs);
      real1d bias("Projection_skip_bias", num_outputs);
      realHost1d weights_flat_h("Projection_skip_weights_h", num_saved * num_outputs);
      realHost1d bias_h("Projection_skip_bias_h", num_outputs);
      for (int i = 0; i < weights_flat_h.extent(0); i++) weights_flat_h(i) = static_cast<real>(data(4 + i));
      for (int i = 0; i < bias_h.extent(0); i++) bias_h(i) = static_cast<real>(data(4 + weights_flat_h.extent(0) + i));
      Kokkos::deep_copy(ponni::flatten(weights), ponni::create_device_copy(weights_flat_h));
      Kokkos::deep_copy(bias, ponni::create_device_copy(bias_h));
      init(weights, bias, trainable);
    }

    void validate(int saved_layer_num_inputs) const {
      if (!params.weights.is_allocated() || !params.bias.is_allocated()) {
        Kokkos::abort("ERROR: Binop_Projection_Add weights/bias not allocated");
      }
      if (params.weights.extent(0) == 0 || params.weights.extent(1) == 0) {
        Kokkos::abort("ERROR: Binop_Projection_Add weights dimensions must be nonzero");
      }
      if (params.weights.extent(1) != params.bias.extent(0)) Kokkos::abort("ERROR: Binop_Projection_Add output size mismatch");
      if (params.weights.extent(0) != saved_layer_num_inputs) {
        Kokkos::abort("ERROR: Binop_Projection_Add saved layer size incompatible with projection weights");
      }
    }
  };

}
