
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N, class real = float>
  struct Binop_Concatenate {
    typedef typename yakl::Array<real,2,yakl::memDevice> real2d;
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = true; // Use two inputs?
    bool static constexpr save            = false;
    int  static constexpr index           = N;

    struct Params {
      int  num_inputs;
      int  num_outputs;
      bool after;
    };

    Params params;

    Binop_Concatenate() {}
    Binop_Concatenate(int num_inputs, int num_outputs, bool after=true) {
      init(num_inputs, num_outputs, after);
    }


    void init(int num_inputs, int num_outputs, bool after=true) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_outputs;
    }


    char const * get_label         () const { return "Binop_Concatenate"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input1, real2d const &input2, real2d const &output,
                                         int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        if (params_in.after) {
          int num_inputs_1 = input1.dimension[0];
          output(irow,ibatch) = irow < num_inputs_1 ? input1(irow,ibatch) : input2(irow - num_inputs_1,ibatch);
        } else {
          int num_inputs_2 = input2.dimension[0];
          output(irow,ibatch) = irow < num_inputs_2 ? input2(irow,ibatch) : input1(irow - num_inputs_2,ibatch);
        }
      }
    }


    void print_verbose() const {
      std::cout << "    concatenating saved index, " << index << ", onto the previous layer's output\n";
    }


    void validate(int saved_layer_num_inputs) const {
      if ( params.num_outputs != saved_layer_num_inputs + params.num_inputs ) {
        yakl::yakl_throw("ERROR: Binop_Concatenate: this layer's num outputs != "
                         "this layer's num inputs + saved layer's num inputs");
      }
    }

  };

}


