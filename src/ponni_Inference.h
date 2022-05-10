
#pragma once
// Included by ponni.h

namespace ponni {



  template <class TUPLE>
  class Inference {
  protected:
    TUPLE layers;
    int static constexpr num_layers = std::tuple_size<TUPLE>::value;

  public:

    Inference(TUPLE const &layers) {
      this->layers = layers;
      int temp_size = get_temporary_size();
      std::cout << "Num layers: " << num_layers << std::endl;
      std::cout << "temp_size: " << temp_size << std::endl;
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


    // Perform inference no this sequential feed-forward model parallelizing only over batches
    real2d batch_parallel( realConst2d input ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      YAKL_SCOPE( layers , this->layers );

      int num_inputs  = std::get<0           >(layers).get_num_inputs ();
      int num_outputs = std::get<num_layers-1>(layers).get_num_outputs();
      int num_batches = input.dimension[1];

      if (input.dimension[0] != std::get<0>(layers).get_num_inputs()) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }

      real2d output("output",num_outputs,num_batches);

      if constexpr (num_layers == 1) {  // Trivial case for one layer
        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          for (int irow = 0; irow < num_outputs; irow++) {
            std::get<0>(layers).compute_one_output( std::get<0>(layers).params , input , output , ibatch , irow );
          }
        });

      } else if constexpr (num_layers == 2) {  // Two or more layers needs either one or two temporary arrays

        real2d tmp("tmp",get_temporary_size(),num_batches);

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          for (int irow = 0; irow < std::get<0>(layers).get_num_outputs(); irow++) {
            std::get<0>(layers).compute_one_output( std::get<0>(layers).params , input , tmp    , ibatch , irow );
          }
          for (int irow = 0; irow < std::get<1>(layers).get_num_outputs(); irow++) {
            std::get<1>(layers).compute_one_output( std::get<1>(layers).params , tmp   , output , ibatch , irow );
          }
        });

      } else {

        int temp_size = get_temporary_size();
        real2d tmp1("tmp1",temp_size,num_batches);
        real2d tmp2("tmp2",temp_size,num_batches);
        bool output_in_tmp1 = false;

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          traverse_layers_batch_parallel(layers,input,output,tmp1,tmp2,output_in_tmp1,ibatch);
        });

      }

      return output;

    } // batch_parallel


    template <int I=0>
    YAKL_INLINE void traverse_layers_batch_parallel(TUPLE const &layers, realConst2d input_glob,
                                                    real2d const &output_glob, real2d const &tmp1, real2d const &tmp2,
                                                    bool output_in_tmp1 , int ibatch) const {
      auto &layer = std::get<I>(layers);
      if constexpr (I == 0) {
        for (int irow = 0; irow < layer.get_num_outputs(); irow++) {
          layer.compute_one_output( layer.params , input_glob , tmp1 , ibatch , irow );
        }
        output_in_tmp1 = true;
        traverse_layers_batch_parallel<I+1>(layers, input_glob, output_glob, tmp1, tmp2, output_in_tmp1, ibatch);
      } else if constexpr (I < num_layers-1) {
        real2d in;
        real2d out;
        if constexpr (layer.overwrite_input) {
          if (output_in_tmp1) { in = tmp1; out = tmp1; }
          else                { in = tmp2; out = tmp2; }
        } else {
          if (output_in_tmp1) { in = tmp1; out = tmp2; output_in_tmp1 = false; }
          else                { in = tmp2; out = tmp1; output_in_tmp1 = true ; }
        }
        for (int irow = 0; irow < layer.get_num_outputs(); irow++) {
          layer.compute_one_output( layer.params , in , out , ibatch , irow );
        }
        traverse_layers_batch_parallel<I+1>(layers, input_glob, output_glob, tmp1, tmp2, output_in_tmp1, ibatch);
      } else {
        real2d in;
        real2d out;
        if (output_in_tmp1) { in = tmp1; }
        else                { in = tmp2; }
        for (int irow = 0; irow < layer.get_num_outputs(); irow++) {
          layer.compute_one_output( layer.params , in , output_glob , ibatch , irow );
        }
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


  };

}


