
#pragma once
// Included by ponni.h

namespace ponni {



  template <class TUPLE>
  class Inference {
  protected:

    TUPLE                layers;
    int static constexpr num_layers = std::tuple_size<TUPLE>::value;

  public:

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

    struct SavedState {
      real2d state;
      int    size;
    };

    typedef typename yakl::SArray<SavedState,1,get_num_saved_states() == 0 ? 1 : get_num_saved_states()> SAVED_TYPE;

    Inference(TUPLE const &layers) {
      this->layers = layers;
    }


    ~Inference() {}


    template <int I=0>
    int get_temporary_size(int max_outputs=0) const {
      if constexpr (I < num_layers-2) {
        return get_temporary_size<I+1>( std::max( std::get<I>(layers).get_num_outputs() , max_outputs ) );
      } else {
        return std::max( std::get<I>(layers).get_num_outputs() , max_outputs );
      }
    }


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


    template <int I=0>
    void allocate_saved_states(SAVED_TYPE &saved_states, int num_batches) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer = std::get<I>(layers);
      if constexpr (I < num_layers) {
        if constexpr (LAYER_TYPE::save) {
          int constexpr index = LAYER_TYPE::index;
          saved_states(index).state = real2d("saved_state",get_saved_state_size<index>(layers),num_batches);
        }
      }
      if constexpr (I < num_layers-1) allocate_saved_states<I+1>( saved_states , num_batches );
    }


    // Perform inference no this sequential feed-forward model parallelizing only over batches
    real2d batch_parallel( realConst2d input ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      YAKL_SCOPE( layers , this->layers );

      auto &layer0     = std::get<0>(layers);
      auto &layer_last = std::get<num_layers-1>(layers);

      int num_inputs  = layer0    .get_num_inputs ();
      int num_outputs = layer_last.get_num_outputs();
      int num_batches = input.dimension[1];

      SAVED_TYPE saved_states;
      allocate_saved_states( saved_states , num_batches );

      if (input.dimension[0] != layer0.get_num_inputs()) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }

      real2d output("output",num_outputs,num_batches);

      if constexpr (num_layers == 1) {  // Trivial case for one layer

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          for (int irow = 0; irow < num_outputs; irow++) {
            layer0.compute_one_output(input, output, ibatch, irow);
          }
        });

      } else {

        int temp_size = get_temporary_size();
        real2d tmp1("tmp1",temp_size,num_batches);
        real2d tmp2("tmp2",temp_size,num_batches);

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          traverse_layers_batch_parallel(layers, saved_states, input, output, tmp1, tmp2, ibatch);
        });

      }

      return output;

    } // batch_parallel


    template <int I=0>
    YAKL_INLINE void static traverse_layers_batch_parallel(TUPLE      const & layers      ,
                                                           SAVED_TYPE const & saved_states,
                                                           realConst2d        input_glob  ,
                                                           real2d     const & output_glob ,
                                                           real2d     const & tmp1        ,
                                                           real2d     const & tmp2        ,
                                                           int                ibatch      ,
                                                           bool               output_in_tmp1 = false) {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &layer       = std::get<I>(layers);
      auto  num_outputs = layer.get_num_outputs();
      realConst2d in;
      real2d      out;
      if constexpr (I == 0) {
        in = input_glob;
        out = tmp1;
        output_in_tmp1 = true;
      } else if constexpr (I < num_layers-1) {
        if constexpr (LAYER_TYPE::overwrite_input) {
          if (output_in_tmp1) { in = tmp1;   out = tmp1; }
          else                { in = tmp2;   out = tmp2; }
        } else {
          if (output_in_tmp1) { in = tmp1;   out = tmp2;   output_in_tmp1 = false; }
          else                { in = tmp2;   out = tmp1;   output_in_tmp1 = true ; }
        }
      } else {
        if (output_in_tmp1) { in = tmp1;   out = output_glob; }
        else                { in = tmp2;   out = output_glob; }
      }

      if constexpr (LAYER_TYPE::save) {
        out = saved_states(LAYER_TYPE::index).state;
        saved_states(LAYER_TYPE::index).size = layer.get_num_inputs();
      }

      if constexpr (LAYER_TYPE::binop) {
        auto &saved = saved_states(LAYER_TYPE::index).state;
        for (int irow = 0; irow < num_outputs; irow++) { layer.compute_one_output(in,saved,out,ibatch,irow); }
      } else {
        for (int irow = 0; irow < num_outputs; irow++) { layer.compute_one_output(in,out,ibatch,irow); }
      }

      if constexpr (I < num_layers-1) {
        traverse_layers_batch_parallel<I+1>(layers,saved_states,input_glob,output_glob,tmp1,tmp2,ibatch,output_in_tmp1);
      }
    }


    template <int I=0>
    void print() const {
      if constexpr (I==0) std::cout << "Inference model has " << num_layers << " layers:\n";
      if constexpr (I < num_layers) {
        std::cout << "  " << std::setw(3) << std::right << I+1 << ": "
                  << std::setw(15) << std::left << std::get<I>(layers).get_label() << " with "
                  << std::get<I>(layers).get_num_inputs() << " inputs and "
                  << std::get<I>(layers).get_num_outputs() << " outputs.\n";
        print<I+1>();
      }
    }


    template <int I=0>
    void print_verbose() const {
      if constexpr (I==0) std::cout << "Inference model has " << num_layers << " layers:\n";
      if constexpr (I < num_layers) {
        std::cout << "  " << std::right << I+1 << ": "
                  << std::left << std::get<I>(layers).get_label() << " with "
                  << std::get<I>(layers).get_num_inputs() << " inputs and "
                  << std::get<I>(layers).get_num_outputs() << " outputs.\n";
        std::get<I>(layers).print_verbose();
        print_verbose<I+1>();
      }
    }


    template <int I = 0>
    void validate( SAVED_TYPE saved_states = SAVED_TYPE() ) const {
      using LAYER_TYPE = typename std::tuple_element<I,TUPLE>::type;
      auto &this_layer = std::get<I>(layers);
      if constexpr (LAYER_TYPE::save) saved_states(LAYER_TYPE::index).size = this_layer.get_num_inputs();
      if constexpr (LAYER_TYPE::binop) {
        int saved_layer_num_inputs = saved_states(LAYER_TYPE::index).size;
        this_layer.validate(saved_layer_num_inputs);
      } else {
        this_layer.validate();
      }
      if constexpr (I < num_layers-1) {
        auto &next_layer = std::get<I+1>(layers);
        if ( this_layer.get_num_outputs() != next_layer.get_num_inputs() ) {
          yakl::yakl_throw("ERROR: This layer's num outputs != next layer's num inputs");
        }
        validate<I+1>(saved_states);
      }
    }


  };

}


