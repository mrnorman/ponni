
#pragma once
// Included by ponni.h

namespace ponni {



  template <class TUPLE>
  class Model {
  protected:
    TUPLE layers;
    int static constexpr num_layers = std::tuple_size<TUPLE>::value;
    real2d tmp1, tmp2;

  public:

    Model(TUPLE const &layers) {
      this->layers = layers;
      int temp_size = get_temporary_size();
      std::cout << "Num layers: " << num_layers << std::endl;
      std::cout << "temp_size: " << temp_size << std::endl;
    }


    ~Model() {}


    template <int I=0>
    int get_temporary_size(int max_outputs=0) const {
      if constexpr (I < num_layers-2) {
        return get_temporary_size<I+1>( std::max( std::get<I>(layers).get_num_outputs() , max_outputs ) );
      } else {
        return std::max( std::get<I>(layers).get_num_outputs() , max_outputs );
      }
    }


    // // Perform inference no this sequential feed-forward model parallelizing only over batches
    // real2d inference_batch_parallel( realConst2d input ) const {
    //   using yakl::c::parallel_for;
    //   using yakl::c::SimpleBounds;

    //   YAKL_SCOPE( layers     , this->layers     );
    //   YAKL_SCOPE( num_layers , this->num_layers );

    //   if (num_layers == 0) yakl::yakl_throw("Error: model is empty");

    //   int num_inputs  = layers(0           ).get_num_inputs ();
    //   int num_outputs = layers(num_layers-1).get_num_outputs();
    //   int num_batches = input.dimension[1];

    //   if (input.dimension[0] != layers(0).get_num_inputs()) {
    //     yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
    //   }

    //   real2d output("output",num_outputs,num_batches);

    //   if (num_layers == 1) {  // Trivial case for one layer

    //     parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
    //       layers(0).apply_batch_parallel( input , output , ibatch );
    //     });

    //   } else {  // Two or more layers needs either one or two temporary arrays

    //     // Get the size of the temporary array(s)
    //     int tmp_size = layers(0).get_num_outputs();
    //     for (int i=1; i < num_layers-1; i++) {
    //       tmp_size = std::max( tmp_size , layers(i).get_num_outputs() );
    //     }

    //     if (num_layers == 2) {  // For two layers, we only need one temporary array

    //       real2d tmp("tmp",tmp_size,num_batches);
    //       parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
    //         layers(0).apply_batch_parallel( input , tmp    , ibatch );
    //         layers(1).apply_batch_parallel( tmp   , output , ibatch );
    //       });

    //     } else {  // For three or more layers, we need two temporary arrays

    //       real2d tmp1("tmp1",tmp_size,num_batches);
    //       real2d tmp2("tmp2",tmp_size,num_batches);
    //       parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
    //         // First layer
    //         layers(0).apply_batch_parallel( input , tmp1 , ibatch );
    //         bool result_in_tmp1 = true;
    //         // Middle layers
    //         for (int i=1; i < num_layers-1; i++) {
    //           if (result_in_tmp1) { layers(i).apply_batch_parallel( tmp1 , tmp2 , ibatch );  result_in_tmp1 = false; }
    //           else                { layers(i).apply_batch_parallel( tmp2 , tmp1 , ibatch );  result_in_tmp1 = true ; }
    //         }
    //         // Last layer
    //         if (result_in_tmp1) { layers(num_layers-1).apply_batch_parallel( tmp1 , output , ibatch ); }
    //         else                { layers(num_layers-1).apply_batch_parallel( tmp2 , output , ibatch ); }
    //       });

    //     } // if (num_layers > 2)

    //   } // if (num_layers > 1)

    //   return output;

    // } // inference_onelevel


    template <int I=0>
    void print() const {
      if constexpr (I==0) std::cout << "Model has " << num_layers << " layers:\n";
      if constexpr (I < num_layers) {
        std::cout << "  " << std::setw(3) << std::right << I+1 << ": "
                  << std::setw(15) << std::left << std::get<I>(layers).get_label() << " with "
                  << std::get<I>(layers).get_num_inputs() << " inputs and "
                  << std::get<I>(layers).get_num_outputs() << " outputs.\n";
        print<I+1>();
      }
    }


    // void print_verbose() {
    //   std::cout << "Model has " << num_layers << " layers:\n";
    //   for (int i=0; i < num_layers; i++) {
    //     std::cout << "  " << i << ": ";
    //     layers(i).print_verbose();
    //   }
    // }


  };

}


