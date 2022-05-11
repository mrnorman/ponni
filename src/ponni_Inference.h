
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

      auto &layer0     = std::get<0>(layers);
      auto &layer_last = std::get<num_layers-1>(layers);

      int num_inputs  = layer0    .get_num_inputs ();
      int num_outputs = layer_last.get_num_outputs();
      int num_batches = input.dimension[1];

      if (input.dimension[0] != layer0.get_num_inputs()) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }

      real2d output("output",num_outputs,num_batches);

      if constexpr (num_layers == 1) {  // Trivial case for one layer

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          for (int irow = 0; irow < num_outputs; irow++) {
            layer0.compute_one_output( layer0.params , input , output , ibatch , irow );
          }
        });

      } else {

        int temp_size = get_temporary_size();
        real2d tmp1("tmp1",temp_size,num_batches);
        real2d tmp2("tmp2",temp_size,num_batches);

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          traverse_layers_batch_parallel(layers, input, output, tmp1, tmp2, ibatch);
        });

      }

      return output;

    } // batch_parallel


    template <int I=0>
    YAKL_INLINE void traverse_layers_batch_parallel(TUPLE const &layers, realConst2d input_glob,
                                                    real2d const &output_glob, real2d const &tmp1, real2d const &tmp2,
                                                    int ibatch, bool output_in_tmp1 = false) const {
      auto &layer       = std::get<I>(layers);
      auto  num_outputs = layer.get_num_outputs();
      auto &params      = layer.params;
      realConst2d in;
      real2d      out;
      if constexpr (I == 0) {
        in = input_glob;   out = tmp1;   output_in_tmp1 = true;
      } else if constexpr (I < num_layers-1) {
        if constexpr (layer.overwrite_input) {
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
      for (int irow = 0; irow < num_outputs; irow++) { layer.compute_one_output(params, in, out, ibatch, irow); }
      if constexpr (I < num_layers-1) {
        traverse_layers_batch_parallel<I+1>(layers, input_glob, output_glob, tmp1, tmp2, ibatch, output_in_tmp1);
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


