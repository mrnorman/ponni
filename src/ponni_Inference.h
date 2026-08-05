
#pragma once
// Included by ponni.h

namespace ponni {



  // Implements a model for inferencing.
  // State: a model state between operations such as input, output, or a "hidden layer"
  // Layer: An *operation* on a state to produce another state such as matrix multiplication
  // The only data held in this class are the saved states and each layer's parameters
  template <class TUPLE,
            class real = float,
            class ExecutionSpace = Kokkos::DefaultExecutionSpace,
            class MemorySpace = typename ExecutionSpace::memory_space>
  struct Inference {
    static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                  "Inference ExecutionSpace must be a Kokkos execution space");
    static_assert(Kokkos::is_memory_space_v<MemorySpace>,
                  "Inference MemorySpace must be a Kokkos memory space");
    static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,MemorySpace>::accessible,
                  "Inference ExecutionSpace cannot access its MemorySpace");

    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    typedef typename Kokkos::View<real   * ,Kokkos::LayoutRight,MemorySpace> real1d;
    typedef typename Kokkos::View<real   **,Kokkos::LayoutRight,MemorySpace> real2d;
    // ***********************************************************************
    // ** FUNCTIONS AND CONSTEXPR VARIABLES NEEDED TO DECLARE CLASS MEMBERS **
    // ***********************************************************************
    int static constexpr num_layers = std::tuple_size<TUPLE>::value;  // Number of layers

    // Get the maximum number of states that need to be in memory at the same time
    template <int I=0>
    int static constexpr get_num_saved_states() {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      if constexpr (I < num_layers-1) {
        if constexpr (LAYER_TYPE::save) { return std::max( LAYER_TYPE::index+1 , get_num_saved_states<I+1>() ); }
        else                            { return                                 get_num_saved_states<I+1>()  ; }
      } else {
        if constexpr (LAYER_TYPE::save) { return LAYER_TYPE::index+1; }
        else                            { return 0;                   }
      }
    }

    // Get the maximum size needed for holding a temporary internal state
    template <int I=0>
    int static constexpr get_max_size_stack() {
      using LAYER_T = typename std::tuple_element_t<I,TUPLE>;
      int constexpr mx = std::max( LAYER_T::INPUT_SIZE , LAYER_T::OUTPUT_SIZE );
      if constexpr (I < num_layers-1) { return std::max( mx , get_max_size_stack<I+1>() ); }
      else                            { return mx; }
    }

    // Return the first layer after a dense operation that cannot be evaluated
    // as part of that dense operation's scalar output loop. A second dense
    // layer must start a new block because all of its inputs are required;
    // normalization, saved-state, binary, and conservative custom layers are
    // barriers for the same reason.
    template <int I>
    int static constexpr fused_dense_block_end() {
      if constexpr (I >= num_layers) {
        return num_layers;
      } else if constexpr (is_pointwise_layer_v<std::tuple_element_t<I,TUPLE>>) {
        return fused_dense_block_end<I + 1>();
      } else {
        return I;
      }
    }

    // Count planned dense blocks for compile-time checks and diagnostics. A
    // barrier does not disable fusion elsewhere in the tuple: traversal simply
    // resumes looking for another dense block after the barrier.
    template <int I=0>
    int static constexpr get_num_fused_dense_blocks() {
      if constexpr (I >= num_layers) {
        return 0;
      } else if constexpr (is_dense_layer_v<std::tuple_element_t<I,TUPLE>>) {
        int constexpr end = fused_dense_block_end<I + 1>();
        return 1 + get_num_fused_dense_blocks<end>();
      } else {
        return get_num_fused_dense_blocks<I + 1>();
      }
    }

    // Mirror the compile-time traversal to size each ping-pong View
    // independently. REQUESTED_BUFFER is zero for tmp1 and one for tmp2;
    // INPUT_IN_TMP1 describes where the state entering layer I resides. The
    // final block writes directly to the caller and therefore consumes no
    // temporary storage.
    template <int REQUESTED_BUFFER, int I=0, bool INPUT_IN_TMP1=false>
    int get_planned_temporary_size(int max_outputs=0) const {
      if constexpr (I >= num_layers) {
        return max_outputs;
      } else if constexpr (is_dense_layer_v<std::tuple_element_t<I,TUPLE>>) {
        int constexpr end = fused_dense_block_end<I + 1>();
        if constexpr (end == num_layers) {
          return max_outputs;
        } else {
          bool constexpr output_in_tmp1 = I == 0 ? true : !INPUT_IN_TMP1;
          int next_max = max_outputs;
          if constexpr (REQUESTED_BUFFER == (output_in_tmp1 ? 0 : 1)) {
            int const size = std::get<I>(params.layers).get_num_outputs();
            next_max = std::max(max_outputs,size);
          }
          return get_planned_temporary_size<REQUESTED_BUFFER,end,output_in_tmp1>(next_max);
        }
      } else if constexpr (I == num_layers - 1) {
        return max_outputs;
      } else {
        using LAYER_TYPE = std::tuple_element_t<I,TUPLE>;
        bool constexpr output_in_tmp1 = I == 0 ? true :
                                             (LAYER_TYPE::overwrite_input ? INPUT_IN_TMP1 : !INPUT_IN_TMP1);
        int next_max = max_outputs;
        if constexpr (REQUESTED_BUFFER == (output_in_tmp1 ? 0 : 1)) {
          int const size = std::get<I>(params.layers).get_num_outputs();
          next_max = std::max(max_outputs,size);
        }
        return get_planned_temporary_size<REQUESTED_BUFFER,I + 1,output_in_tmp1>(next_max);
      }
    }

    // A saved state is used for things like ResNet or DenseNet
    // "size" might seem redundant, but the size of the state is large enough to hold the *largest* state and
    //    may not be indicative of the actual size. Think of this as a partially filled array pattern
    struct SavedState {
      real2d state;
      int    size;
    };

    // This declares an ponni::SArray type that holds SavedState inner types to hold all necessary saved states at a given time
    typedef typename ponni::SArray<SavedState,get_num_saved_states() == 0 ? 1 : get_num_saved_states()> SAVED_TYPE;

    // ****************************************************
    // ** ALL DATA MEMBERS ARE INSIDE THIS PARAMS STRUCT **
    // ****************************************************
    // This must be passed to the static in-kernel inferencing function as a parameter
    // The batch_parallel inferencing does not need this passed as a parameter, though
    struct Params {
      SAVED_TYPE  saved_states;  // For holding states saved for later binary operations (ResNet, DenseNet, etc.)
      TUPLE       layers;        // The operations performed successively on the input
      real2d      tmp1;          // For alternating storage of temporary states while traversing the model
      real2d      tmp2;          // For alternating storage of temporary states while traversing the model
    };

    Params params;
    ExecutionSpace execution_space_;
    int internal_state_capacity_ = 0;
 
    // **********************************
    // ** BEGIN CLASS MEMBER FUNCTIONS **
    // **********************************

    Inference() = default;
    ~Inference() = default;

    // This is not intended to be called directly by the user per se. It's easier to call ponni::create_inference_model
    Inference(TUPLE const & layers, ExecutionSpace const & execution_space = ExecutionSpace())
      : execution_space_(execution_space) {
      this->params.layers = layers;
    }



    // Reallocate every batch-dependent internal view to exactly batch_size.
    // Ordinary batch inference grows this storage automatically; this public
    // method is useful for shrinking or releasing retained capacity.
    void reallocate_internal_state(int batch_size) {
      if (batch_size < 0) Kokkos::abort("Inference internal-state batch size must be nonnegative");
      execution_space_.fence("PONNI internal-state reallocation");
      reallocate_saved_states(batch_size);
      reallocate_temporary(params.tmp1, get_planned_temporary_size<0>(), batch_size);
      reallocate_temporary(params.tmp2, get_planned_temporary_size<1>(), batch_size);
      internal_state_capacity_ = batch_size;
    }



    int internal_state_capacity() const { return internal_state_capacity_; }



    // Avoid retaining a meaningless (0,batch) allocation when block planning
    // proves that a ping-pong buffer is unused.
    void reallocate_temporary(real2d & temporary, int feature_size, int batch_size) {
      if (feature_size == 0 || batch_size == 0) {
        temporary = real2d();
      } else {
        Kokkos::realloc(Kokkos::view_alloc(execution_space_, Kokkos::WithoutInitializing),
                        temporary, feature_size, batch_size);
      }
    }


    // Grow on demand, but retain larger allocations for later calls.
    void ensure_internal_state_capacity(int batch_size) {
      if (batch_size > internal_state_capacity()) reallocate_internal_state(batch_size);
    }



    // Get the maximum size needed for a given saved state. All saved states will be allocated at this size
    template <int INDEX, int I=0>
    int get_saved_state_size( TUPLE const &layers ) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>( layers );
      if constexpr (I < num_layers-1) {
        if constexpr (LAYER_TYPE::save) {
          if constexpr (LAYER_TYPE::index == INDEX) {
            return std::max( get_saved_state_size<INDEX,I+1>(layers) , layer.get_num_outputs() );
          }
        }
        return std::max( get_saved_state_size<INDEX,I+1>(layers) , 0 );
      } else {
        if constexpr (LAYER_TYPE::save) {
          if constexpr (LAYER_TYPE::index == INDEX) {
            return layer.get_num_outputs();
          }
        }
        return 0;
      }
    }



    // Resize every saved residual state alongside the two traversal buffers.
    template <int I=0>
    void reallocate_saved_states(int batch_size) {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      if constexpr (I < num_layers) {
        if constexpr (LAYER_TYPE::save) {
          int constexpr index = LAYER_TYPE::index;
          Kokkos::realloc(Kokkos::view_alloc(execution_space_, Kokkos::WithoutInitializing),
                          params.saved_states(index).state,
                          get_saved_state_size<index>(params.layers), batch_size);
        }
      }
      if constexpr (I < num_layers-1) reallocate_saved_states<I+1>(batch_size);
    }



    // Get the total number of trainable parameters in the model
    template <int I=0>
    int get_num_trainable_parameters() const {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-1) {
        return layer.get_num_trainable_parameters() + get_num_trainable_parameters<I+1>();
      } else {
        return layer.get_num_trainable_parameters();
      }
    }



    template <int I>
    decltype(std::get<I>(params.layers)) & get_layer() { return std::get<I>(params.layers); }



    int get_num_inputs () const { return std::get<0           >(params.layers).get_num_inputs (); }
    int get_num_outputs() const { return std::get<num_layers-1>(params.layers).get_num_outputs(); }



    // Identify the exact ordered tuple that consumes a flattened trainable-
    // parameter tensor. Weight values are intentionally excluded so another
    // trained instance of the same model remains compatible.
    template <int I=0>
    void append_weight_schema(std::ostringstream & schema) const {
      auto const & layer = std::get<I>(params.layers);
      schema << I << '\t' << layer.get_label() << '\t'
             << layer.get_num_inputs() << '\t' << layer.get_num_outputs() << '\t'
             << layer.get_num_trainable_parameters() << '\n';
      if constexpr (I < num_layers - 1) append_weight_schema<I + 1>(schema);
    }



    std::string weight_schema_fingerprint() const {
      std::ostringstream schema;
      schema << "ponni-template-model-v1\n";
      append_weight_schema(schema);
      std::string const text = schema.str();
      auto const * bytes = reinterpret_cast<unsigned char const *>(text.data());
      return ponni_fnv1a64_string(ponni_fnv1a64(bytes,text.size()));
    }



    // Load the canonical flattened parameter tensor only after the PONNI file
    // has passed its structural, checksum, schema, dtype, shape, and exact
    // templated-model fingerprint checks.
    bool load_weights(std::string const & path, std::string * error = nullptr) {
      int const parameter_count = get_num_trainable_parameters();
      if (parameter_count == 0) {
        if (error != nullptr) *error = "cannot load weights into a model with no trainable parameters";
        return false;
      }
      PonniFile file;
      if (!file.load(path,error)) return false;
      std::string const dtype = std::is_same_v<real,double> ? "F64" : "F32";
      std::vector<PonniTensorSpec> const expected{{"parameters",dtype,{static_cast<std::size_t>(parameter_count)},0}};
      if (!file.validate(expected,weight_schema_fingerprint(),error)) return false;
      auto const * tensor = file.find("parameters");
      unsigned char const * bytes = file.tensor_data(*tensor);
      Kokkos::View<real*,Kokkos::LayoutRight,Kokkos::HostSpace> host("ponni_template_parameters_host",parameter_count);
      if constexpr (std::is_same_v<real,double>) {
        for (int i = 0; i < parameter_count; i++) host(i) = detail::read_scalar<double>(bytes + 8 * i);
      } else {
        for (int i = 0; i < parameter_count; i++) host(i) = static_cast<real>(detail::read_scalar<float>(bytes + 4 * i));
      }
      set_trainable_parameters(create_memory_space_copy(host,MemorySpace()));
      return true;
    }



    bool save_weights(std::string const & path, std::string * error = nullptr) const {
      int const parameter_count = get_num_trainable_parameters();
      if (parameter_count == 0) {
        if (error != nullptr) *error = "cannot save weights from a model with no trainable parameters";
        return false;
      }
      auto const parameters_host = create_host_copy(get_trainable_parameters());
      using StoredScalar = std::conditional_t<std::is_same_v<real,double>,double,float>;
      std::string const dtype = std::is_same_v<StoredScalar,double> ? "F64" : "F32";
      std::vector<PonniTensorSpec> const specs{{"parameters",dtype,{static_cast<std::size_t>(parameter_count)},0}};
      std::vector<StoredScalar> stored(static_cast<std::size_t>(parameter_count));
      for (int i = 0; i < parameter_count; i++) stored[i] = static_cast<StoredScalar>(parameters_host(i));
      return write_ponni_file(path,specs,weight_schema_fingerprint(),stored.data(),error,"template");
    }



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    template <class InputView>
    real2d forward_batch_parallel(InputView const & input) {
      ponni::require_layout_right_views<InputView>();
      int const batch_size = static_cast<int>(input.extent(1));
      if (batch_size == 0) Kokkos::abort("Inference requires a nonzero batch size");
      real2d output("output", get_num_outputs(), batch_size);
      forward_batch_parallel(input, output);
      return output;
    }



    // Execute into caller-provided storage. Views may use any memory space the
    // selected execution space can access; PONNI owns only retained scratch.
    template <class InputView, class OutputView>
    void forward_batch_parallel(InputView const & input, OutputView const & output) {
      ponni::require_layout_right_views<InputView,OutputView>();
      static_assert(Kokkos::is_view_v<InputView> && InputView::rank == 2,
                    "Inference input must be a rank-two Kokkos::View");
      static_assert(Kokkos::is_view_v<OutputView> && OutputView::rank == 2,
                    "Inference output must be a rank-two Kokkos::View");
      static_assert(!std::is_const_v<typename OutputView::value_type>,
                    "Inference output View must be writable");
      static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,typename InputView::memory_space>::accessible,
                    "Inference ExecutionSpace cannot access the input View");
      static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,typename OutputView::memory_space>::accessible,
                    "Inference ExecutionSpace cannot access the output View");
      int const batch_size = static_cast<int>(input.extent(1));
      if (batch_size == 0) Kokkos::abort("Inference requires a nonzero batch size");
      ensure_internal_state_capacity(batch_size);
      PONNI_SCOPE( layers       , this->params.layers       );
      PONNI_SCOPE( saved_states , this->params.saved_states );
      PONNI_SCOPE( tmp1         , this->params.tmp1         );
      PONNI_SCOPE( tmp2         , this->params.tmp2         );
      auto &layer0      = std::get<0>(layers);
      auto &layer_last  = std::get<num_layers-1>(layers);
      if (input.extent(0) != layer0.get_num_inputs()) {
        Kokkos::abort("Error: Provided # inputs differs from model's # inputs");
      }
      if (output.extent(0) != layer_last.get_num_outputs() || output.extent(1) != input.extent(1)) {
        Kokkos::abort("Error: Provided output dimensions do not match the model and input batch");
      }
      // Every tuple uses this one kernel launch. The recursive mixed executor
      // below is entirely compile-time: it fuses legal dense/pointwise regions,
      // materializes barriers, and erases unused buffer-selection branches in
      // each concrete model instantiation.
      Kokkos::parallel_for(PONNI_AUTO_LABEL(),
                           Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, batch_size),
                           KOKKOS_LAMBDA(int ibatch) {
        traverse_mixed_batch_parallel(layers, saved_states, input, output, tmp1, tmp2, ibatch);
      });
    } // forward_batch_parallel



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void forward_batch_parallel_in_kernel(InputView const & input,
                                                                        OutputView const & output,
                                                                        Params const & params_in,
                                                                        int ibatch) {
      ponni::require_layout_right_views<InputView,OutputView>();
      auto &layer0 = std::get<0>(params_in.layers);
      #ifdef PONNI_DEBUG
        if (input.extent(0) != layer0.get_num_inputs(layer0.params)) {
          Kokkos::abort("Error: Provided # inputs differs from model's # inputs");
        }
      #endif
      traverse_mixed_batch_parallel(params_in.layers, params_in.saved_states, input, output,
                                    params_in.tmp1, params_in.tmp2, ibatch);
    } // forward_batch_parallel_in_kernel



    // Apply the pointwise layers following a dense layer to a register scalar.
    // Since BEGIN and END are compile-time tuple indices, the compiler sees a
    // direct sequence of calls rather than runtime dispatch.
    template <int BEGIN, int END>
    KOKKOS_INLINE_FUNCTION static real apply_fused_epilogue(real value, int feature, TUPLE const & layers) {
      if constexpr (BEGIN < END) {
        using LAYER_TYPE = std::tuple_element_t<BEGIN,TUPLE>;
        auto const & layer = std::get<BEGIN>(layers);
        value = apply_fused_layer<LAYER_TYPE>(value, feature, layer.params);
        return apply_fused_epilogue<BEGIN + 1,END>(value, feature, layers);
      } else {
        return value;
      }
    }



    // Compute a dense layer and all of its pointwise epilogues in one output
    // loop. Only the fully transformed value is written to a View.
    template <int BEGIN, int END, class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void compute_fused_dense_block(TUPLE const & layers,
                                                                 InputView const & input,
                                                                 OutputView const & output,
                                                                 int ibatch) {
      using DENSE_TYPE = std::tuple_element_t<BEGIN,TUPLE>;
      auto const & dense = std::get<BEGIN>(layers);
      int const num_outputs = dense.get_num_outputs(dense.params);
      for (int feature = 0; feature < num_outputs; feature++) {
        real value = DENSE_TYPE::compute_output(input, feature, ibatch, dense.params);
        value = apply_fused_epilogue<BEGIN + 1,END>(value, feature, layers);
        output(feature,ibatch) = value;
      }
    }



    // Execute a mixed plan inside one batch-parallel kernel. Dense layers and
    // their consecutive pointwise epilogues form fused blocks; every other
    // layer is a materialization barrier. INPUT_IN_TMP1 is a template argument,
    // not runtime state, so each instantiation contains only the input/output
    // View selections it actually uses.
    template <int I=0, bool INPUT_IN_TMP1=false, class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void traverse_mixed_batch_parallel(TUPLE const & layers,
                                                                     SAVED_TYPE const & saved_states,
                                                                     InputView const & input_glob,
                                                                     OutputView const & output_glob,
                                                                     real2d const & tmp1,
                                                                     real2d const & tmp2,
                                                                     int ibatch) {
      using LAYER_TYPE = std::tuple_element_t<I,TUPLE>;
      auto const & layer = std::get<I>(layers);

      if constexpr (is_dense_layer_v<LAYER_TYPE>) {
        // Fold only the immediately following pointwise operations into this
        // dense output loop. If this is the final block, bypass temporary
        // storage and write the caller's output directly.
        int constexpr end = fused_dense_block_end<I + 1>();
        if constexpr (I == 0) {
          if constexpr (end == num_layers) {
            compute_fused_dense_block<I,end>(layers, input_glob, output_glob, ibatch);
          } else {
            compute_fused_dense_block<I,end>(layers, input_glob, tmp1, ibatch);
            traverse_mixed_batch_parallel<end,true>(layers, saved_states, input_glob, output_glob,
                                                    tmp1, tmp2, ibatch);
          }
        } else {
          real2d in;
          if constexpr (INPUT_IN_TMP1) in = tmp1;
          else                         in = tmp2;

          if constexpr (end == num_layers) {
            compute_fused_dense_block<I,end>(layers, in, output_glob, ibatch);
          } else {
            bool constexpr output_in_tmp1 = !INPUT_IN_TMP1;
            real2d out;
            if constexpr (output_in_tmp1) out = tmp1;
            else                          out = tmp2;
            compute_fused_dense_block<I,end>(layers, in, out, ibatch);
            traverse_mixed_batch_parallel<end,output_in_tmp1>(layers, saved_states,
                                                               input_glob, output_glob,
                                                               tmp1, tmp2, ibatch);
          }
        }
      } else if constexpr (I == 0 && num_layers == 1) {
        // Preserve the direct single-layer path without instantiating either
        // temporary View or any saved/binary machinery that cannot be used.
        layer.compute_all_outputs(input_glob, output_glob, ibatch, layer.params);
      } else if constexpr (I == 0) {
        // A leading barrier follows the historical rule: materialize into
        // tmp1, except for a layer whose contract explicitly saves its output.
        if constexpr (LAYER_TYPE::save) {
          saved_states(LAYER_TYPE::index).size = layer.get_num_inputs(layer.params);
          // Save_State is an identity on the main path. A leading save has no
          // previously materialized main-path buffer, so populate both the
          // residual slot and tmp1 before traversal continues from tmp1.
          layer.compute_all_outputs(input_glob, saved_states(LAYER_TYPE::index).state,
                                    ibatch, layer.params);
          layer.compute_all_outputs(input_glob, tmp1, ibatch, layer.params);
        } else {
          layer.compute_all_outputs(input_glob, tmp1, ibatch, layer.params);
        }
        traverse_mixed_batch_parallel<I + 1,true>(layers, saved_states, input_glob, output_glob,
                                                   tmp1, tmp2, ibatch);
      } else if constexpr (I < num_layers - 1) {
        // Barrier inputs and destinations are selected at compile time. An
        // in-place barrier preserves the active buffer; other barriers toggle
        // it. Save_State redirects the write to its named residual View while
        // deliberately leaving the main-path buffer selection unchanged.
        real2d in;
        if constexpr (INPUT_IN_TMP1) in = tmp1;
        else                         in = tmp2;

        bool constexpr output_in_tmp1 = LAYER_TYPE::overwrite_input ?
                                             INPUT_IN_TMP1 : !INPUT_IN_TMP1;
        real2d out;
        if constexpr (output_in_tmp1) out = tmp1;
        else                          out = tmp2;

        if constexpr (LAYER_TYPE::save) {
          out = saved_states(LAYER_TYPE::index).state;
          saved_states(LAYER_TYPE::index).size = layer.get_num_inputs(layer.params);
        }
        if constexpr (LAYER_TYPE::binop) {
          auto const & saved = saved_states(LAYER_TYPE::index).state;
          layer.compute_all_outputs(in, saved, out, ibatch, layer.params);
        } else {
          layer.compute_all_outputs(in, out, ibatch, layer.params);
        }
        traverse_mixed_batch_parallel<I + 1,output_in_tmp1>(layers, saved_states,
                                                            input_glob, output_glob,
                                                            tmp1, tmp2, ibatch);
      } else {
        // The last barrier also bypasses temporary output storage.
        real2d in;
        if constexpr (INPUT_IN_TMP1) in = tmp1;
        else                         in = tmp2;
        if constexpr (LAYER_TYPE::binop) {
          auto const & saved = saved_states(LAYER_TYPE::index).state;
          layer.compute_all_outputs(in, saved, output_glob, ibatch, layer.params);
        } else {
          layer.compute_all_outputs(in, output_glob, ibatch, layer.params);
        }
      }
    } // traverse_mixed_batch_parallel



    int static constexpr IN_GL  = std::tuple_element_t<0           ,TUPLE>::INPUT_SIZE;
    int static constexpr OUT_GL = std::tuple_element_t<num_layers-1,TUPLE>::OUTPUT_SIZE;

    // SArray inference keeps residual states in per-thread stack storage. Each
    // slot has the exact maximum compile-time width required by its save index,
    // allowing models with differently sized simultaneous residuals.
    template <int INDEX, int I=0>
    int static constexpr get_static_saved_state_size() {
      using LAYER_TYPE = std::tuple_element_t<I,TUPLE>;
      if constexpr (LAYER_TYPE::save) {
        if constexpr (LAYER_TYPE::index == INDEX) {
          if constexpr (I < num_layers - 1) {
            return std::max(LAYER_TYPE::OUTPUT_SIZE,get_static_saved_state_size<INDEX,I + 1>());
          } else {
            return LAYER_TYPE::OUTPUT_SIZE;
          }
        } else {
          if constexpr (I < num_layers - 1) return get_static_saved_state_size<INDEX,I + 1>();
          else                              return 0;
        }
      } else {
        if constexpr (I < num_layers - 1) return get_static_saved_state_size<INDEX,I + 1>();
        else                              return 0;
      }
    }

    template <std::size_t... Indices>
    static auto make_local_saved_state_type(std::index_sequence<Indices...>)
        -> std::tuple<ponni::SArray<real,get_static_saved_state_size<static_cast<int>(Indices)>()>...>;

    using LOCAL_SAVED_TYPE = decltype(make_local_saved_state_type(
        std::make_index_sequence<get_num_saved_states()>{}));

    KOKKOS_INLINE_FUNCTION static void forward_batch_parallel_in_kernel( ponni::SArray<real,IN_GL > const & input     ,
                                                                         ponni::SArray<real,OUT_GL>       & output    ,
                                                                         Params                     const & params_in ) {
      if constexpr (num_layers == 1) {
        auto &layer0 = std::get<0>(params_in.layers);
        layer0.compute_all_outputs(input,output,layer0.params);
      } else {
        LOCAL_SAVED_TYPE saved_states;
        ponni::SArray<real,std::tuple_element_t<0,TUPLE>::OUTPUT_SIZE> tmp;
        traverse_layers_batch_parallel(params_in.layers, saved_states, input, output,
                                       ponni::SArray<real,IN_GL>(), tmp);
      }
    } // forward_batch_parallel_in_kernel



    // Traverse the layers of this model inside a GPU kernel
    template <int I = 0>
    KOKKOS_INLINE_FUNCTION void static traverse_layers_batch_parallel(
        TUPLE const & layers,
        LOCAL_SAVED_TYPE & saved_states,
        ponni::SArray<real,IN_GL> const & in_glob,
        ponni::SArray<real,OUT_GL> & out_glob,
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::INPUT_SIZE> const & in,
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::OUTPUT_SIZE> & out) {
      auto &layer = std::get<I>(layers);
      if constexpr (I == 0) {
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::OUTPUT_SIZE> tmp;
        if constexpr (std::tuple_element_t<I,TUPLE>::save) {
          layer.compute_all_outputs(in_glob,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),layer.params);
          layer.compute_all_outputs(in_glob,tmp,layer.params);
        } else if constexpr (std::tuple_element_t<I,TUPLE>::binop) {
          layer.compute_all_outputs(in_glob,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),
                                    tmp,layer.params);
        } else {
          layer.compute_all_outputs(in_glob,tmp,layer.params);
        }
        if constexpr (std::tuple_element_t<I+1,TUPLE>::overwrite_input &&
                      std::tuple_element_t<I+1,TUPLE>::INPUT_SIZE ==
                          std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE) {
          traverse_layers_batch_parallel<I+1>(layers, saved_states, in_glob, out_glob, tmp, tmp);
        } else {
          ponni::SArray<real,std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE> tmp2;
          traverse_layers_batch_parallel<I+1>(layers, saved_states, in_glob, out_glob, tmp, tmp2);
        }
      } else if constexpr (I < num_layers-1) {
        if constexpr (std::tuple_element_t<I,TUPLE>::save) {
          layer.compute_all_outputs(in,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),layer.params);
          layer.compute_all_outputs(in,out,layer.params);
        } else if constexpr (std::tuple_element_t<I,TUPLE>::binop) {
          layer.compute_all_outputs(in,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),out,layer.params);
        } else {
          layer.compute_all_outputs(in,out,layer.params);
        }
        if constexpr (std::tuple_element_t<I+1,TUPLE>::overwrite_input &&
                      std::tuple_element_t<I+1,TUPLE>::INPUT_SIZE ==
                          std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE) {
          traverse_layers_batch_parallel<I+1>(layers, saved_states, in_glob, out_glob, out, out);
        } else {
          ponni::SArray<real,std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE> tmp;
          traverse_layers_batch_parallel<I+1>(layers, saved_states, in_glob, out_glob, out, tmp);
        }
      } else {
        if constexpr (std::tuple_element_t<I,TUPLE>::save) {
          layer.compute_all_outputs(in,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),layer.params);
          layer.compute_all_outputs(in,out_glob,layer.params);
        } else if constexpr (std::tuple_element_t<I,TUPLE>::binop) {
          layer.compute_all_outputs(in,std::get<std::tuple_element_t<I,TUPLE>::index>(saved_states),
                                    out_glob,layer.params);
        } else {
          layer.compute_all_outputs(in,out_glob,layer.params);
        }
      }
    }



    // Print basic information about this model
    template <int I=0>
    void print() const {
      if constexpr (I==0) std::cout << "Inference model has " << num_layers << " layers -- with "
                                    << get_num_trainable_parameters() << " total trainable parameters.\n";
      if constexpr (I < num_layers) {
        auto &layer = std::get<I>(params.layers);
        std::cout << "  " << std::setw(3) << std::right << I+1 << ": "
                  << std::setw(15) << std::left << layer.get_label() << " with "
                  << layer.get_num_inputs () << " inputs, " << layer.get_num_outputs() << " outputs, and "
                  << layer.get_num_trainable_parameters() << " trainable parameters\n";
        print<I+1>();
      }
    }



    // Distribute a flattened parameter View to the layer tuple in declaration
    // order. Each layer consumes the prefix matching its trainable count.
    template <int I=0>
    void set_trainable_parameters(real1d in) {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-1) {
        layer.set_trainable_parameters(in);
        std::size_t const consumed = static_cast<std::size_t>(layer.get_num_trainable_parameters());
        in = Kokkos::subview(in,std::make_pair(consumed,in.extent(0)));
        set_trainable_parameters<I+1>(in);
      } else  {
        layer.set_trainable_parameters(in);
      }
    }



    // Gather the tuple's parameters into the same flattened declaration order.
    template <int I=0>
    real1d get_trainable_parameters(real1d params_glob = real1d() , int offset = 0) const {
      if constexpr (I == 0) params_glob = real1d("params_glob",get_num_trainable_parameters());
      auto params_loc = std::get<I>(params.layers).get_trainable_parameters();
      if (params_loc.is_allocated()) {
        auto arr = Kokkos::subview(params_glob,std::make_pair(
            static_cast<std::size_t>(offset),static_cast<std::size_t>(offset + params_loc.size())));
        Kokkos::deep_copy(arr,params_loc);
        offset += params_loc.size();
      }
      if constexpr (I < num_layers-1) { return get_trainable_parameters<I+1>( params_glob , offset ); }
      else                            { return params_glob; }
    }

    // Validate that the input and output sizes of each layer match up
    template <int I = 0>
    void validate( SAVED_TYPE saved_states = SAVED_TYPE() ) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &this_layer = std::get<I>(params.layers);
      if constexpr (LAYER_TYPE::save) saved_states(LAYER_TYPE::index).size = this_layer.get_num_inputs();
      if constexpr (LAYER_TYPE::binop) {
        int saved_layer_num_inputs = saved_states(LAYER_TYPE::index).size;
        this_layer.validate(saved_layer_num_inputs);
      } else {
        this_layer.validate();
      }
      if constexpr (I < num_layers-1) {
        auto &next_layer = std::get<I+1>(params.layers);
        if ( this_layer.get_num_outputs() != next_layer.get_num_inputs() ) {
          Kokkos::abort("ERROR: This layer's num outputs != next layer's num inputs");
        }
        validate<I+1>(saved_states);
      }
    }


  };

}
