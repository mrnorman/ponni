
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
    typedef typename Kokkos::View<double * ,Kokkos::LayoutRight,Kokkos::HostSpace > doubleHost1d;
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

    // Get the maximum size needed for holding a temporary internal state
    template <int I=0>
    int get_temporary_size(int max_outputs=0) const {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-2) {
        return get_temporary_size<I+1>( std::max( layer.get_num_outputs() , max_outputs ) );
      } else {
        return std::max( layer.get_num_outputs() , max_outputs );
      }
    }

    // A tuple is eligible for the optimized dynamic-View path only when it
    // starts with a dense layer and every remaining operation is either dense
    // or pointwise. Barriers use the established general traversal unchanged.
    template <int I=0>
    bool static constexpr is_fused_feed_forward() {
      using LAYER_TYPE = std::tuple_element_t<I,TUPLE>;
      if constexpr (I == 0 && !is_dense_layer_v<LAYER_TYPE>) {
        return false;
      } else if constexpr (!is_dense_layer_v<LAYER_TYPE> && !is_pointwise_layer_v<LAYER_TYPE>) {
        return false;
      } else if constexpr (I + 1 < num_layers) {
        return is_fused_feed_forward<I + 1>();
      } else {
        return true;
      }
    }

    // Locate the next dense layer at compile time. All intervening operations
    // in an eligible tuple are pointwise epilogues of the preceding dense.
    template <int I>
    int static constexpr next_dense_layer() {
      if constexpr (I >= num_layers) {
        return num_layers;
      } else if constexpr (is_dense_layer_v<std::tuple_element_t<I,TUPLE>>) {
        return I;
      } else {
        return next_dense_layer<I + 1>();
      }
    }

    // Return the largest materialized dense-block output assigned to one of
    // the two ping-pong buffers. The final block writes directly to the caller,
    // so a one-block model needs no temporary View and a two-block model needs
    // only tmp1.
    template <int I=0, int BUFFER=0>
    int get_fused_temporary_size(int requested_buffer) const {
      int constexpr next_dense = next_dense_layer<I + 1>();
      if constexpr (next_dense == num_layers) {
        return 0;
      } else {
        auto const & dense = std::get<I>(params.layers);
        int const this_size = BUFFER == requested_buffer ? dense.get_num_outputs() : 0;
        return std::max(this_size, get_fused_temporary_size<next_dense,1 - BUFFER>(requested_buffer));
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
      if constexpr (is_fused_feed_forward()) {
        reallocate_temporary(params.tmp1, get_fused_temporary_size(0), batch_size);
        reallocate_temporary(params.tmp2, get_fused_temporary_size(1), batch_size);
      } else {
        reallocate_temporary(params.tmp1, get_temporary_size(), batch_size);
        reallocate_temporary(params.tmp2, get_temporary_size(), batch_size);
      }
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



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    template <class InputView>
    real2d forward_batch_parallel(InputView const & input) {
      int const batch_size = static_cast<int>(input.extent(1));
      real2d output("output", get_num_outputs(), batch_size);
      forward_batch_parallel(input, output);
      return output;
    }



    // Execute into caller-provided storage. Views may use any memory space the
    // selected execution space can access; PONNI owns only retained scratch.
    template <class InputView, class OutputView>
    void forward_batch_parallel(InputView const & input, OutputView const & output) {
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
      if constexpr (is_fused_feed_forward()) {
        Kokkos::parallel_for(PONNI_AUTO_LABEL(),
                             Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, batch_size),
                             KOKKOS_LAMBDA(int ibatch) {
          traverse_fused_feed_forward(layers, input, output, tmp1, tmp2, ibatch);
        });
      } else if constexpr (num_layers == 1) {  // Trivial case for one layer
        Kokkos::parallel_for(PONNI_AUTO_LABEL(),
                             Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, batch_size),
                             KOKKOS_LAMBDA(int ibatch) {
          layer0.compute_all_outputs(input, output, ibatch, layer0.params);
        });
      } else {
        Kokkos::parallel_for(PONNI_AUTO_LABEL(),
                             Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, batch_size),
                             KOKKOS_LAMBDA(int ibatch) {
          traverse_layers_batch_parallel(layers, saved_states, input, output, tmp1, tmp2, ibatch);
        });
      }
    } // forward_batch_parallel



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void forward_batch_parallel_in_kernel(InputView const & input,
                                                                        OutputView const & output,
                                                                        Params const & params_in,
                                                                        int ibatch) {
      auto &layer0 = std::get<0>(params_in.layers);
      #ifdef PONNI_DEBUG
        if (input.extent(0) != layer0.get_num_inputs(layer0.params)) {
          Kokkos::abort("Error: Provided # inputs differs from model's # inputs");
        }
      #endif
      if constexpr (is_fused_feed_forward()) {
        traverse_fused_feed_forward(params_in.layers, input, output, params_in.tmp1, params_in.tmp2, ibatch);
      } else if constexpr (num_layers == 1) {
        layer0.compute_all_outputs(input,output,ibatch,layer0.params);
      } else {
        traverse_layers_batch_parallel(params_in.layers,params_in.saved_states,input,output,
                                       params_in.tmp1,params_in.tmp2,ibatch);
      }
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



    // Traverse materialized dense blocks rather than individual layers. Dense
    // outputs alternate between tmp1 and tmp2; the last block bypasses scratch
    // and writes directly to the application-provided output View.
    template <int BEGIN=0, class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION static void traverse_fused_feed_forward(TUPLE const & layers,
                                                                    InputView const & input,
                                                                    OutputView const & output,
                                                                    real2d const & tmp1,
                                                                    real2d const & tmp2,
                                                                    int ibatch,
                                                                    bool write_tmp1=true) {
      int constexpr next_dense = next_dense_layer<BEGIN + 1>();
      if constexpr (next_dense == num_layers) {
        compute_fused_dense_block<BEGIN,num_layers>(layers, input, output, ibatch);
      } else {
        real2d const next_state = write_tmp1 ? tmp1 : tmp2;
        compute_fused_dense_block<BEGIN,next_dense>(layers, input, next_state, ibatch);
        traverse_fused_feed_forward<next_dense>(layers, next_state, output, tmp1, tmp2, ibatch, !write_tmp1);
      }
    }



    // Traverse the layers of this model inside a GPU kernel
    template <int I=0, class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION void static traverse_layers_batch_parallel(TUPLE const & layers,
                                                                      SAVED_TYPE const & saved_states,
                                                                      InputView const & input_glob,
                                                                      OutputView const & output_glob,
                                                                      real2d const & tmp1,
                                                                      real2d const & tmp2,
                                                                      int ibatch,
                                                                      bool output_in_tmp1 = false) {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>(layers);

      // Only the first and last operations touch application-owned Views.
      // All intermediate operations use model-owned Views in MemorySpace.
      if constexpr (I == 0) {
        real2d out = tmp1;
        if constexpr (LAYER_TYPE::save) {
          out = saved_states(LAYER_TYPE::index).state;
          saved_states(LAYER_TYPE::index).size = layer.get_num_inputs(layer.params);
        }
        layer.compute_all_outputs(input_glob, out, ibatch, layer.params);
        output_in_tmp1 = true;
      } else if constexpr (I < num_layers-1) {
        real2d in;
        real2d out;
        if constexpr (LAYER_TYPE::overwrite_input) {
          if (output_in_tmp1) { in = tmp1;   out = tmp1; }
          else                { in = tmp2;   out = tmp2; }
        } else {
          if (output_in_tmp1) { in = tmp1;   out = tmp2;   output_in_tmp1 = false; }
          else                { in = tmp2;   out = tmp1;   output_in_tmp1 = true ; }
        }

        if constexpr (LAYER_TYPE::save) {
          out = saved_states(LAYER_TYPE::index).state;
          saved_states(LAYER_TYPE::index).size = layer.get_num_inputs(layer.params);
        }
        if constexpr (LAYER_TYPE::binop) {
          auto &saved = saved_states(LAYER_TYPE::index).state;
          layer.compute_all_outputs(in, saved, out, ibatch, layer.params);
        } else {
          layer.compute_all_outputs(in, out, ibatch, layer.params);
        }
      } else {
        real2d const in = output_in_tmp1 ? tmp1 : tmp2;
        if constexpr (LAYER_TYPE::binop) {
          auto &saved = saved_states(LAYER_TYPE::index).state;
          layer.compute_all_outputs(in, saved, output_glob, ibatch, layer.params);
        } else {
          layer.compute_all_outputs(in, output_glob, ibatch, layer.params);
        }
      }

      if constexpr (I < num_layers-1) {
        traverse_layers_batch_parallel<I+1>(layers, saved_states, input_glob, output_glob,
                                            tmp1, tmp2, ibatch, output_in_tmp1);
      }
    } // traverse_layers_batch_parallel



    int static constexpr IN_GL  = std::tuple_element_t<0           ,TUPLE>::INPUT_SIZE;
    int static constexpr OUT_GL = std::tuple_element_t<num_layers-1,TUPLE>::OUTPUT_SIZE;
    KOKKOS_INLINE_FUNCTION static void forward_batch_parallel_in_kernel( ponni::SArray<real,IN_GL > const & input     ,
                                                                         ponni::SArray<real,OUT_GL>       & output    ,
                                                                         Params                     const & params_in ) {
      if constexpr (num_layers == 1) {
        auto &layer0 = std::get<0>(params_in.layers);
        layer0.compute_all_outputs(input,output,layer0.params);
      } else {
        ponni::SArray<real,std::tuple_element_t<0,TUPLE>::OUTPUT_SIZE> tmp;
        traverse_layers_batch_parallel( params_in.layers , input , output , ponni::SArray<real,IN_GL>() , tmp );
      }
    } // forward_batch_parallel_in_kernel



    // Traverse the layers of this model inside a GPU kernel
    template <int I = 0>
    KOKKOS_INLINE_FUNCTION void static traverse_layers_batch_parallel(
        TUPLE const & layers,
        ponni::SArray<real,IN_GL> const & in_glob,
        ponni::SArray<real,OUT_GL> & out_glob,
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::INPUT_SIZE> const & in,
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::OUTPUT_SIZE> & out) {
      auto &layer = std::get<I>(layers);
      if constexpr (I == 0) {
        ponni::SArray<real,std::tuple_element_t<I,TUPLE>::OUTPUT_SIZE> tmp;
        layer.compute_all_outputs(in_glob,tmp,layer.params);
        if constexpr (std::tuple_element_t<I+1,TUPLE>::overwrite_input) {
          traverse_layers_batch_parallel<I+1>( layers , in_glob , out_glob , tmp , tmp );
        } else {
          ponni::SArray<real,std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE> tmp2;
          traverse_layers_batch_parallel<I+1>( layers , in_glob , out_glob , tmp , tmp2 );
        }
      } else if constexpr (I < num_layers-1) {
        layer.compute_all_outputs(in,out,layer.params);
        if constexpr (std::tuple_element_t<I+1,TUPLE>::overwrite_input) {
          traverse_layers_batch_parallel<I+1>( layers , in_glob , out_glob , out , out );
        } else {
          ponni::SArray<real,std::tuple_element_t<I+1,TUPLE>::OUTPUT_SIZE> tmp;
          traverse_layers_batch_parallel<I+1>( layers , in_glob , out_glob , out , tmp );
        }
      } else {
        layer.compute_all_outputs(in,out_glob,layer.params);
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



    // Set the model layers' trainable parameters. Input dimensioned as (num_parameters,num_ensembles)
    template <int I=0>
    void set_trainable_parameters(real1d in) {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-1) {
        layer.set_trainable_parameters(in);
        in = in.subset_slowest_dimension(layer.get_num_trainable_parameters(),in.extent(0)-1);
        set_trainable_parameters<I+1>(in);
      } else  {
        layer.set_trainable_parameters(in);
      }
    }



    // Set the model layers' trainable parameters. Input dimensioned as (num_parameters,num_ensembles)
    template <int I=0>
    real1d get_trainable_parameters(real1d params_glob = real1d() , int offset = 0) const {
      if constexpr (I == 0) params_glob = real1d("params_glob",get_num_trainable_parameters());
      auto params_loc = std::get<I>(params.layers).get_trainable_parameters();
      if (params_loc.is_allocated()) {
        auto arr = params_glob.subset_slowest_dimension(offset,offset+params_loc.size()-1);
        params_loc.deep_copy_to(arr);
        offset += params_loc.size();
      }
      if constexpr (I < num_layers-1) { return get_trainable_parameters<I+1>( params_glob , offset ); }
      else                            { return params_glob; }
    }




    // Get the total number of double precision elements needed to store this model in a flattened array representation
    template <int I=0>
    int get_array_representation_size() const {
      auto sz = std::get<I>(params.layers).get_array_representation_size();
      if constexpr (I < num_layers-1) return sz + get_array_representation_size<I+1>();
      else                            return sz;
    }



    // Represent this model as a flattened Host-memory double precision array
    template <int I=0>
    doubleHost1d represent_as_array( doubleHost1d array = doubleHost1d() , int offset = 0 ) const {
      if constexpr (I == 0) array = doubleHost1d("model_as_array",get_array_representation_size());
      auto tmp = std::get<I>(params.layers).to_array();
      for (int i=0; i < tmp.size(); i++) { array(offset+i) = tmp(i); }
      if constexpr (I < num_layers-1) return represent_as_array<I+1>( array , offset + tmp.size() );
      else                            return array;
    }



    // Set the layer parameters from a flattened array representation
    template <int I=0>
    void set_layers_from_array_representation( doubleHost1d const &array ) {
      std::get<I>(params.layers).from_array(array);
      int offset = std::get<I>(params.layers).get_array_representation_size();
      if (offset > array.size()) Kokkos::abort("ERROR: Incompatible array representation");
      doubleHost1d tmp( array.data()+offset , array.size()-offset );
      if constexpr (I < num_layers-1) set_layers_from_array_representation<I+1>(tmp);
    }



    template <int I=0>
    void save_to_text_file( std::string fname , std::ofstream file = std::ofstream() ) {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I == 0) {
        file.open(fname);
        file << "number_of_layers: " << num_layers << "\n";
        file << "layer_types_listed_below:\n";
        file << layer.get_label() << "\n";
        save_to_text_file<I+1>( fname , std::move(file) );
      } else if constexpr (I < num_layers-1) {
        file << layer.get_label() << "\n"; 
        save_to_text_file<I+1>( fname , std::move(file) );
      } else {
        file << layer.get_label() << "\n";
        auto array = represent_as_array();
        file << "number_of_elements_in_flattened_representation: " << array.size() << "\n";
        file << "flattened_representation_below_one_line_per_value: \n";
        for (int i=0; i < array.size(); i++) { file << std::setprecision(17) << array(i) << "\n"; }
        file.close();
      }
    }



    template <int I=0>
    void load_from_text_file( std::string fname , std::ifstream file = std::ifstream() ) {
      auto &layer = std::get<I>(params.layers);
      std::string dummy;
      if constexpr (I == 0) {
        file.open(fname);
        if (! file.is_open()) { std::cerr << "ERROR: Failed to open " << fname << std::endl; Kokkos::abort(""); }
        int file_num_layers;  file >> dummy >> file_num_layers;
        if (file_num_layers != num_layers) { Kokkos::abort("ERROR: Incorrect number of layers in saved file"); }
        file >> dummy;
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { Kokkos::abort("ERROR: Incorrect layer type"); }
        load_from_text_file<I+1>( fname , std::move(file) );
      } else if constexpr (I < num_layers-1) {
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { Kokkos::abort("ERROR: Incorrect layer type"); }
        load_from_text_file<I+1>( fname , std::move(file) );
      } else {
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { Kokkos::abort("ERROR: Incorrect layer type"); }
        int num_flattened_values;  file >> dummy >> num_flattened_values;
        doubleHost1d array("flattened_representation",num_flattened_values);
        file >> dummy;
        for (int i=0; i < num_flattened_values; i++) { file >> array(i); }
        set_layers_from_array_representation( array );
        file.close();
      }
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
