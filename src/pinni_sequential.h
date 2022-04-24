
#pragma once

#include "pinni_layer.h"

namespace pinni {

  template <int MAX_LAYERS>
  class Sequential {
  protected:
    yakl::SArray<Layer,1,MAX_LAYERS> layers;
    int                              num_layers;

    void copy_data(Sequential const &rhs) {
      this->num_layers = rhs.num_layers;
      this->layers     = rhs.layers    ;
    }

  public:

    Sequential() { num_layers = 0; }

    ~Sequential() {
      for (int i=0; i < num_layers; i++) { layers(i) = Layer(); }
      num_layers = 0; 
    }

    YAKL_INLINE Sequential                  (Sequential const  &rhs) { copy_data(rhs); }
    YAKL_INLINE Sequential                  (Sequential const &&rhs) { copy_data(rhs); }
    YAKL_INLINE Sequential const & operator=(Sequential const  &rhs) { if (this == &rhs) return *this; copy_data(rhs); return *this; }
    YAKL_INLINE Sequential const & operator=(Sequential const &&rhs) { if (this == &rhs) return *this; copy_data(rhs); return *this; }


    void add_layer( Layer const &layer ) {
      auto msg = layer.validate();  if (msg != "") yakl::yakl_throw(msg.c_str());
      if ( num_layers > 0 ) {
        if ( layers(num_layers-1).get_num_outputs() != layer.get_num_inputs() ) {
          yakl::yakl_throw("Error: previous layer's # inputs != this layer's # outputs");
        }
      }
      if (num_layers == MAX_LAYERS) {
        yakl::yakl_throw("Error: Trying to add too many layers. Please increase MAX_LAYERS template parameter");
      }
      layers(num_layers) = layer;
      num_layers++;
    }


    // Perform inference no this sequential feed-forward model parallelizing only over batches
    real2d inference_batchparallel( realConst2d input ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      YAKL_SCOPE( layers     , this->layers     );
      YAKL_SCOPE( num_layers , this->num_layers );

      if (num_layers == 0) yakl::yakl_throw("Error: model is empty");

      int num_inputs  = input.dimension[0];
      int num_batches = input.dimension[1];
      int num_outputs = layers(num_layers-1).get_num_outputs();

      if (num_inputs != layers(0).get_num_inputs()) {
        yakl::yakl_throw("Error: Provided # inputs differs from model's # inputs");
      }

      real2d output("output",num_outputs,num_batches);

      if (num_layers == 1) {  // Trivial case for one layer

        parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
          layers(0).apply_serial( input , output , ibatch );
        });

      } else {  // Two or more layers needs either one or two temporary arrays

        // Get the size of the temporary array(s)
        int tmp_size = layers(0).get_num_outputs();
        for (int i=1; i < num_layers-1; i++) {
          tmp_size = std::max( tmp_size , layers(i).get_num_outputs() );
        }

        if (num_layers == 2) {  // For two layers, we only need one temporary array

          real2d tmp("tmp",tmp_size,num_batches);
          parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
            layers(0).apply_serial( input , tmp    , ibatch );
            layers(1).apply_serial( tmp   , output , ibatch );
          });

        } else {  // For three or more layers, we need two temporary arrays

          real2d tmp1("tmp1",tmp_size,num_batches);
          real2d tmp2("tmp2",tmp_size,num_batches);
          parallel_for( SimpleBounds<1>(num_batches) , YAKL_LAMBDA (int ibatch) {
            // First layer
            layers(0).apply_serial( input , tmp1 , ibatch );
            bool result_in_tmp1 = true;
            // Middle layers
            for (int i=1; i < num_layers-1; i++) {
              if (result_in_tmp1) { layers(i).apply_serial( tmp1 , tmp2 , ibatch );  result_in_tmp1 = false; }
              else                { layers(i).apply_serial( tmp2 , tmp1 , ibatch );  result_in_tmp1 = true ; }
            }
            // Last layer
            if (result_in_tmp1) { layers(num_layers-1).apply_serial( tmp1 , output , ibatch ); }
            else                { layers(num_layers-1).apply_serial( tmp2 , output , ibatch ); }
          });

        } // if (num_layers > 2)

      } // if (num_layers > 1)

      return output;

    } // inference_onelevel


    Layer get_last_layer() { return layers(num_layers-1); }


    void print() {
      std::cout << "Sequential model has " << num_layers << " layers:\n";
      for (int i=0; i < num_layers; i++) {
        std::cout << "  " << std::setw(3) << std::right << i+1 << ": "
                  << std::setw(15) << std::left << layers(i).get_type_str() << " with "
                  << layers(i).get_num_inputs() << " inputs and "
                  << layers(i).get_num_outputs() << " outputs.\n";
      }
    }


    void print_verbose() {
      std::cout << "Sequential model has " << num_layers << " layers:\n";
      for (int i=0; i < num_layers; i++) {
        std::cout << "  " << i << ": ";
        layers(i).print_verbose();
      }
    }


  };

}


