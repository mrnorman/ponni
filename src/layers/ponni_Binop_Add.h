
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N>
  class Binop_Add {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = N;

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Binop_Add() {}
    Binop_Add(int num_inputs) {
      init(num_inputs);
    }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    char const * get_label      () const { return "Binop_Add"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE void compute_one_output(realConst2d input1, realConst2d input2,
                                        real2d const &output, int ibatch, int irow) const {
      output(irow,ibatch) = input1(irow,ibatch) + input2(irow,ibatch);
    }


    void print_verbose() const {
      std::cout << "    adding from saved index: " << index << "\n";
    }


    void validate(int saved_layer_num_inputs) const {
      if ( params.num_inputs != saved_layer_num_inputs ) {
        yakl::yakl_throw("ERROR: Binop_Add: Saved layer num inputs != this layer's num inputs");
      }
    }

  };

}


