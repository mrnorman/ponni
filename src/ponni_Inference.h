
#pragma once
// Included by ponni.h

namespace ponni {



  // Implements a model for inferencing.
  // State: a model state between operations such as input, output, or a "hidden layer"
  // Layer: An *operation* on a state to produce another state such as matrix multiplication
  // The only data held in this class are the saved states and each layer's parameters
  template <class TUPLE, class real = float>
  struct Inference {
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;
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
    int constexpr get_temporary_size(int max_outputs=0) const {
      if constexpr (I < num_layers-2) {
        return get_temporary_size<I+1>( std::max( std::get<I>(params.layers).get_num_outputs(std::get<I>(params.layers).params) ,
                                                  max_outputs ) );
      } else {
        return std::max( std::get<I>(params.layers).get_num_outputs(std::get<I>(params.layers).params) , max_outputs );
      }
    }

    // A saved state is used for things like ResNet or DenseNet
    // "size" might seem redundant, but the size of the state is large enough to hold the *largest* state and
    //    may not be indicative of the actual size. Think of this as a partially filled array pattern
    struct SavedState {
      real2d state;
      int    size;
    };

    // This declares an SArray type that holds SavedState inner types to hold all necessary saved states at a given time
    typedef typename yakl::SArray<SavedState,1,get_num_saved_states() == 0 ? 1 : get_num_saved_states()> SAVED_TYPE;

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
 
    // **********************************
    // ** BEGIN CLASS MEMBER FUNCTIONS **
    // **********************************

    Inference() = default;
    ~Inference() = default;

    // This is not intended to be called directly by the user per se. It's easier to call ponni::create_inference_model
    Inference(TUPLE const &layers, int num_batches = 1) {
      this->params.layers = layers;
      allocate_saved_states(num_batches);
      params.tmp1 = real2d("ponni_tmp1",get_temporary_size(),num_batches);
      params.tmp2 = real2d("ponni_tmp2",get_temporary_size(),num_batches);
    }



    // Set the batch size to allocate arrays to hold saved and temporary states. Mainly used for in-kernel inferencing
    void set_batch_size(int num_batches) {
      allocate_saved_states(num_batches);
      params.tmp1 = real2d("ponni_tmp1",get_temporary_size(),num_batches);
      params.tmp2 = real2d("ponni_tmp2",get_temporary_size(),num_batches);
    }



    // Get the maximum size needed for a given saved state. All saved states will be allocated at this size
    template <int INDEX, int I=0>
    int get_saved_state_size( TUPLE const &layers ) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>( layers );
      if constexpr (I < num_layers-1) {
        if constexpr (LAYER_TYPE::save) {
          if constexpr (LAYER_TYPE::index == INDEX) {
            return std::max( get_saved_state_size<INDEX,I+1>(layers) , layer.get_num_outputs(layer.params) );
          }
        }
        return std::max( get_saved_state_size<INDEX,I+1>(layers) , 0 );
      } else {
        if constexpr (LAYER_TYPE::save) {
          if constexpr (LAYER_TYPE::index == INDEX) {
            return layer.get_num_outputs(layer.params);
          }
        }
        return 0;
      }
    }



    // Allocate all saved states at their appropriate sizes
    // This is called by the model traversal, not by the user directly
    template <int I=0>
    void allocate_saved_states(int num_batches) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers) {
        if constexpr (LAYER_TYPE::save) {
          int constexpr index = LAYER_TYPE::index;
          params.saved_states(index).state = real2d("saved_state",get_saved_state_size<index>(params.layers),num_batches);
        }
      }
      if constexpr (I < num_layers-1) allocate_saved_states<I+1>( num_batches );
    }



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    real2d forward_batch_parallel( real2d const &input ) const {
      auto layers       = this->params.layers      ;
      auto saved_states = this->params.saved_states;
      auto tmp1         = this->params.tmp1        ;
      auto tmp2         = this->params.tmp2        ;
      // Get the number of inputs, outputs, and batch size
      auto &layer0     = std::get<0>(layers);
      auto &layer_last = std::get<num_layers-1>(layers);
      int num_inputs  = layer0    .get_num_inputs (layer0    .params);
      int num_outputs = layer_last.get_num_outputs(layer_last.params);
      int num_batches = input.dimension[1];
      // Allocate the saved states (overrides default allocation for one batch in constructor)
      allocate_saved_states( num_batches );
      // Ensure input dimension is correct
      if (input.dimension[0] != layer0.get_num_inputs(layer0.params)) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }
      // Allocate the output array
      real2d output("output",num_outputs,num_batches);
      if constexpr (num_layers == 1) {  // Trivial case for one layer
        // GPU kernel threading over batches
        yakl::c::parallel_for( YAKL_AUTO_LABEL() , num_batches , YAKL_LAMBDA (int ibatch) {
          layer0.compute_all_outputs(input, output, ibatch, layer0.params);
        });
      } else {
        // We'll need to store the temporary states in alternating arrays
        // This overrides the default allocate of batch size of one
        int temp_size = get_temporary_size();
        tmp1 = real2d("tmp1",temp_size,num_batches);
        tmp2 = real2d("tmp2",temp_size,num_batches);
        // GPU kernel threading over batch size that traverses the model's layers
        yakl::c::parallel_for( YAKL_AUTO_LABEL() , num_batches , YAKL_LAMBDA (int ibatch) {
          traverse_layers(layers, saved_states, input, output, tmp1, tmp2, ibatch);
        });
      }
      return output;
    } // forward_batch_parallel



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    YAKL_INLINE static void forward_in_kernel( real2d const &input , real2d const &output , Params const &params_in ,
                                               int ibatch ) {
      auto &layer0 = std::get<0>(params_in.layers);
      #ifdef PONNI_DEBUG
        if (input.dimension[0] != layer0.get_num_inputs(layer0.params)) {
          yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
        }
      #endif
      if constexpr (num_layers == 1) {
        layer0.compute_all_outputs(input,output,ibatch,layer0.params);
      } else {
        traverse_layers(params_in.layers,params_in.saved_states,input,output,params_in.tmp1,params_in.tmp2,ibatch);
      }
    } // forward_in_kernel



    // Traverse the layers of this model inside a GPU kernel
    template <int I=0>
    YAKL_INLINE void static traverse_layers(TUPLE      const & layers      ,
                                            SAVED_TYPE const & saved_states,
                                            real2d     const & input_glob  ,
                                            real2d     const & output_glob ,
                                            real2d     const & tmp1        ,
                                            real2d     const & tmp2        ,
                                            int                ibatch      ,
                                            bool               output_in_tmp1 = false) {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer       = std::get<I>(layers);
      auto  num_outputs = layer.get_num_outputs(layer.params);
      // These are placeholder arrays to point to the appropriate tempoarary array, input, or output
      real2d in;
      real2d out;
      if constexpr (I == 0) {
        // First layer has global input as the input array and tmp1 as the output
        in = input_glob;
        out = tmp1;
        output_in_tmp1 = true;
      } else if constexpr (I < num_layers-1) {
        if constexpr (LAYER_TYPE::overwrite_input) {
          // If we can overwrite the input, then input and output will be the same tmp array
          if (output_in_tmp1) { in = tmp1;   out = tmp1; }
          else                { in = tmp2;   out = tmp2; }
        } else {
          // Otherwise, the input is the current tmp array, and the output is the other one
          if (output_in_tmp1) { in = tmp1;   out = tmp2;   output_in_tmp1 = false; }
          else                { in = tmp2;   out = tmp1;   output_in_tmp1 = true ; }
        }
      } else {
        // If this is the last layer, then output is the global output array
        if (output_in_tmp1) { in = tmp1;   out = output_glob; }
        else                { in = tmp2;   out = output_glob; }
      }
      // If this is a "save" layer, then save the current state into the declared index of the saved state arrays
      if constexpr (LAYER_TYPE::save) {
        out = saved_states(LAYER_TYPE::index).state;
        saved_states(LAYER_TYPE::index).size = layer.get_num_inputs(layer.params);
      }
      if constexpr (LAYER_TYPE::binop) {
        // If this is a binary operator (meaning an operation of the current state against a saved state), then get the
        //    correct saved state and perform the requested operation (usually addition or concatenation)
        auto &saved = saved_states(LAYER_TYPE::index).state;
        layer.compute_all_outputs(in,saved,out,ibatch,layer.params);
      } else {
        // Otherwise, apply this layer to the current state to produce the next state
        layer.compute_all_outputs(in,out,ibatch,layer.params);
      }
      // If this isn't the last layer, then call the next layer recursively with template recursion
      if constexpr (I < num_layers-1) {
        traverse_layers<I+1>(layers,saved_states,input_glob,output_glob,tmp1,tmp2,ibatch,output_in_tmp1);
      }
    } // travers_layers



    // Print basic information about this model
    template <int I=0>
    void print() const {
      if constexpr (I==0) std::cout << "Inference model has " << num_layers << " layers:\n";
      if constexpr (I < num_layers) {
        std::cout << "  " << std::setw(3) << std::right << I+1 << ": "
                  << std::setw(15) << std::left << std::get<I>(params.layers).get_label() << " with "
                  << std::get<I>(params.layers).get_num_inputs (std::get<I>(params.layers).params) << " inputs and "
                  << std::get<I>(params.layers).get_num_outputs(std::get<I>(params.layers).params) << " outputs.\n";
        print<I+1>();
      }
    }



    // Print detailed information about this model
    template <int I=0>
    void print_verbose() const {
      if constexpr (I==0) std::cout << "Inference model has " << num_layers << " layers:\n";
      if constexpr (I < num_layers) {
        std::cout << "  " << std::right << I+1 << ": "
                  << std::left << std::get<I>(params.layers).get_label() << " with "
                  << std::get<I>(params.layers).get_num_inputs (std::get<I>(params.layers).params) << " inputs and "
                  << std::get<I>(params.layers).get_num_outputs(std::get<I>(params.layers).params) << " outputs.\n";
        std::get<I>(params.layers).print_verbose();
        print_verbose<I+1>();
      }
    }



    // Validate that the input and output sizes of each layer match up
    template <int I = 0>
    void validate( SAVED_TYPE saved_states = SAVED_TYPE() ) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &this_layer = std::get<I>(params.layers);
      if constexpr (LAYER_TYPE::save) saved_states(LAYER_TYPE::index).size = this_layer.get_num_inputs(this_layer.params);
      if constexpr (LAYER_TYPE::binop) {
        int saved_layer_num_inputs = saved_states(LAYER_TYPE::index).size;
        this_layer.validate(saved_layer_num_inputs);
      } else {
        this_layer.validate();
      }
      if constexpr (I < num_layers-1) {
        auto &next_layer = std::get<I+1>(params.layers);
        if ( this_layer.get_num_outputs(this_layer.params) != next_layer.get_num_inputs(next_layer.params) ) {
          yakl::yakl_throw("ERROR: This layer's num outputs != next layer's num inputs");
        }
        validate<I+1>(saved_states);
      }
    }


  };

}


