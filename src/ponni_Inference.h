
#pragma once
// Included by ponni.h

namespace ponni {



  // Implements a model for inferencing.
  // State: a model state between operations such as input, output, or a "hidden layer"
  // Layer: An *operation* on a state to produce another state such as matrix multiplication
  // The only data held in this class are the saved states and each layer's parameters
  template <class TUPLE, class real = float>
  struct Inference {
    typedef typename yakl::Array<double,1,yakl::memHost  > doubleHost1d;
    typedef typename yakl::Array<real  ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real  ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real  ,3,yakl::memDevice> real3d;
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
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-2) {
        return get_temporary_size<I+1>( std::max( layer.get_num_outputs() , max_outputs ) );
      } else {
        return std::max( layer.get_num_outputs() , max_outputs );
      }
    }

    // A saved state is used for things like ResNet or DenseNet
    // "size" might seem redundant, but the size of the state is large enough to hold the *largest* state and
    //    may not be indicative of the actual size. Think of this as a partially filled array pattern
    struct SavedState {
      real3d state;
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
      real3d      tmp1;          // For alternating storage of temporary states while traversing the model
      real3d      tmp2;          // For alternating storage of temporary states while traversing the model
    };

    Params params;
 
    // **********************************
    // ** BEGIN CLASS MEMBER FUNCTIONS **
    // **********************************

    Inference() = default;
    ~Inference() = default;

    // This is not intended to be called directly by the user per se. It's easier to call ponni::create_inference_model
    Inference(TUPLE const &layers, int batch_size = 1, int num_ensembles = 1) {
      this->params.layers = layers;
      init(batch_size,num_ensembles);
    }



    // Set the batch size to allocate arrays to hold saved and temporary states. Mainly used for in-kernel inferencing
    void init(int batch_size, int num_ensembles) {
      allocate_saved_states(batch_size,num_ensembles);
      params.tmp1 = real3d("ponni_tmp1",get_temporary_size(),batch_size,num_ensembles);
      params.tmp2 = real3d("ponni_tmp2",get_temporary_size(),batch_size,num_ensembles);
    }



    int get_batch_size() const { return params.tmp1.extent(1); }



    int get_num_ensembles() const { return params.tmp1.extent(2); }


    
    bool initialized() const { return params.tmp1.initialized(); }



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



    // Allocate all saved states at their appropriate sizes
    // This is called by the model traversal, not by the user directly
    template <int I=0>
    void allocate_saved_states(int batch_size, int num_ensembles) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers) {
        if constexpr (LAYER_TYPE::save) {
          int constexpr index = LAYER_TYPE::index;
          params.saved_states(index).state = real3d("saved_state",get_saved_state_size<index>(params.layers),
                                                                  batch_size,num_ensembles);
        }
      }
      if constexpr (I < num_layers-1) allocate_saved_states<I+1>( batch_size , num_ensembles );
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



    real2d forward_batch_parallel( real2d const &input ) {
      auto output = forward_batch_parallel(input.reshape(input.extent(0),input.extent(1),1));
      return output.reshape(output.extent(0),output.extent(1));
    }



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    real3d forward_batch_parallel( real3d const &input ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto layers       = this->params.layers      ;
      auto saved_states = this->params.saved_states;
      auto tmp1         = this->params.tmp1        ;
      auto tmp2         = this->params.tmp2        ;
      // Get the number of inputs, outputs, and batch size
      auto &layer0      = std::get<0>(layers);
      auto &layer_last  = std::get<num_layers-1>(layers);
      int num_inputs    = layer0    .get_num_inputs ();
      int num_outputs   = layer_last.get_num_outputs();
      int batch_size    = input.extent(1);
      int num_ensembles = input.extent(2);
      // Allocate the saved states (overrides default allocation for one batch in constructor)
      init( batch_size , num_ensembles );
      // Ensure input dimension is correct
      if (input.extent(0) != layer0.get_num_inputs()) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }
      // Allocate the output array
      real3d output("output",num_outputs,batch_size,num_ensembles);
      if constexpr (num_layers == 1) {  // Trivial case for one layer
        // GPU kernel threading over batches
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(batch_size,num_ensembles) ,
                                          YAKL_LAMBDA (int ibatch, int iens) {
          layer0.compute_all_outputs(input, output, ibatch, iens, layer0.params);
        });
      } else {
        // We'll need to store the temporary states in alternating arrays
        // This overrides the default allocate of batch size of one
        int temp_size = get_temporary_size();
        // GPU kernel threading over batch size that traverses the model's layers
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(batch_size,num_ensembles) ,
                                          YAKL_LAMBDA (int ibatch, int iens) {
          traverse_layers(layers, saved_states, input, output, tmp1, tmp2, ibatch, iens);
        });
      }
      return output;
    } // forward_batch_parallel



    // Perform a forward inference pass through this model parallelizing only the batch dimension
    YAKL_INLINE static void forward_in_kernel( real3d const &input , real3d const &output , Params const &params_in ,
                                               int ibatch , int iens ) {
      auto &layer0 = std::get<0>(params_in.layers);
      #ifdef PONNI_DEBUG
        if (input.extent(0) != layer0.get_num_inputs(layer0.params)) {
          yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
        }
      #endif
      if constexpr (num_layers == 1) {
        layer0.compute_all_outputs(input,output,ibatch,iens,layer0.params);
      } else {
        traverse_layers(params_in.layers,params_in.saved_states,input,output,params_in.tmp1,params_in.tmp2,ibatch,iens);
      }
    } // forward_in_kernel



    // Traverse the layers of this model inside a GPU kernel
    template <int I=0>
    YAKL_INLINE void static traverse_layers(TUPLE      const & layers      ,
                                            SAVED_TYPE const & saved_states,
                                            real3d     const & input_glob  ,
                                            real3d     const & output_glob ,
                                            real3d     const & tmp1        ,
                                            real3d     const & tmp2        ,
                                            int                ibatch      ,
                                            int                iens        ,
                                            bool               output_in_tmp1 = false) {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer       = std::get<I>(layers);
      auto  num_outputs = layer.get_num_outputs(layer.params);
      // These are placeholder arrays to point to the appropriate tempoarary array, input, or output
      real3d in;
      real3d out;
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
        layer.compute_all_outputs(in,saved,out,ibatch,iens,layer.params);
      } else {
        // Otherwise, apply this layer to the current state to produce the next state
        layer.compute_all_outputs(in,out,ibatch,iens,layer.params);
      }
      // If this isn't the last layer, then call the next layer recursively with template recursion
      if constexpr (I < num_layers-1) {
        traverse_layers<I+1>(layers,saved_states,input_glob,output_glob,tmp1,tmp2,ibatch,iens,output_in_tmp1);
      }
    } // travers_layers



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
    void set_trainable_parameters(real2d in) {
      auto &layer = std::get<I>(params.layers);
      if constexpr (I < num_layers-1) {
        layer.set_trainable_parameters(in,false);  // No fence
        in = in.subset_slowest_dimension(layer.get_num_trainable_parameters(),in.extent(0)-1);
        set_trainable_parameters<I+1>(in);
      } else  {
        layer.set_trainable_parameters(in,true);   // Fence
      }
    }



    // Set the model layers' trainable parameters. Input dimensioned as (num_parameters,num_ensembles)
    template <int I=0>
    real2d get_trainable_parameters(real1d params_glob = real1d() , int offset = 0) const {
      if constexpr (I == 0) params_glob = real1d("params_glob",get_num_trainable_parameters()*get_num_ensembles());
      auto params_loc = std::get<I>(params.layers).get_trainable_parameters();
      if (params_loc.initialized()) {
        auto arr = params_glob.subset_slowest_dimension(offset,offset+params_loc.size()-1);
        params_loc.deep_copy_to(arr);
        offset += params_loc.size();
      }
      if constexpr (I < num_layers-1) { return get_trainable_parameters<I+1>( params_glob , offset ); }
      else                            { return params_glob.reshape(get_num_trainable_parameters(),get_num_ensembles()); }
    }



    template <int I=0>
    real1d get_lbounds( real1d lbounds = real1d() , int offset = 0 ) {
      if constexpr (I == 0) lbounds = real1d("lbounds",get_num_trainable_parameters());
      auto &layer = std::get<I>(params.layers);
      auto lbounds_layer = layer.get_lbounds();
      if (lbounds_layer.initialized()) {
        lbounds_layer.deep_copy_to(lbounds.subset_slowest_dimension(offset,offset+lbounds_layer.size()-1));
        offset += lbounds_layer.size();
      }
      if constexpr (I < num_layers-1) { return get_lbounds<I+1>( lbounds , offset ); }
      else                            { return lbounds; }
    }



    template <int I=0>
    real1d get_ubounds( real1d ubounds = real1d() , int offset = 0 ) {
      if constexpr (I == 0) ubounds = real1d("ubounds",get_num_trainable_parameters());
      auto &layer = std::get<I>(params.layers);
      auto ubounds_layer = layer.get_ubounds();
      if (ubounds_layer.initialized()) {
        ubounds_layer.deep_copy_to(ubounds.subset_slowest_dimension(offset,offset+ubounds_layer.size()-1));
        offset += ubounds_layer.size();
      }
      if constexpr (I < num_layers-1) { return get_ubounds<I+1>( ubounds , offset ); }
      else                            { return ubounds; }
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
      if (offset > array.size()) yakl::yakl_throw("ERROR: Incompatible array representation");
      doubleHost1d tmp( array.label() , array.data()+offset , array.size()-offset );
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
        if (! file.is_open()) { std::cerr << "ERROR: Failed to open " << fname << std::endl; yakl::yakl_throw(""); }
        int file_num_layers;  file >> dummy >> file_num_layers;
        if (file_num_layers != num_layers) { yakl::yakl_throw("ERROR: Incorrect number of layers in saved file"); }
        file >> dummy;
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { yakl::yakl_throw("ERROR: Incorrect layer type"); }
        load_from_text_file<I+1>( fname , std::move(file) );
      } else if constexpr (I < num_layers-1) {
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { yakl::yakl_throw("ERROR: Incorrect layer type"); }
        load_from_text_file<I+1>( fname , std::move(file) );
      } else {
        std::string file_layer_label;  file >> file_layer_label;
        if (file_layer_label != layer.get_label()) { yakl::yakl_throw("ERROR: Incorrect layer type"); }
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
          yakl::yakl_throw("ERROR: This layer's num outputs != next layer's num inputs");
        }
        validate<I+1>(saved_states);
      }
    }


  };

}


