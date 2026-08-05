#pragma once
// Included by ponni.h

namespace ponni {

  template <int ISAVE,
            class real = float,
            int NIN = 1,
            int NSAVE = 1,
            class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  struct Binop_Projection_Add {
    using memory_space = MemorySpace;
    template <class NewMemorySpace> using rebind_memory_space = Binop_Projection_Add<ISAVE,real,NIN,NSAVE,NewMemorySpace>;
    typedef Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;

    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true;
    bool static constexpr save            = false;

    // Projection-add reads a second saved vector and performs its own dense
    // operation, so it is a fusion barrier. Custom branch-merging layers should
    // remain barriers until their data dependencies are explicitly planned.
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;
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

    // Copy both parameter Views when the model factory selects a new memory
    // space. Configuration fields remain ordinary constructor arguments.
    template <class NewMemorySpace>
    auto copy_to_memory_space(NewMemorySpace const & memory_space = NewMemorySpace()) const {
      return rebind_memory_space<NewMemorySpace>(
          ponni::create_memory_space_copy(params.weights, memory_space),
          ponni::create_memory_space_copy(params.bias, memory_space), params.trainable);
    }

    char const * get_label() const { return "Binop_Projection_Add"; }
    KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & params_in) { return params_in.weights.extent(1); }
    KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & params_in) { return params_in.weights.extent(1); }
    int get_num_inputs() const { return params.weights.extent(1); }
    int get_num_outputs() const { return params.weights.extent(1); }
    int get_num_trainable_parameters() const { return params.trainable ? params.weights.size() + params.bias.size() : 0; }

    template <class InputView1, class InputView2, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView1 const & input,
                                                           InputView2 const & saved,
                                                           OutputView const & output,
                                                           int ibatch,
                                                           Params const & params_in) {
      ponni::require_layout_right_views<InputView1,InputView2,OutputView>();
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
