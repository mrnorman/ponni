
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N>
  class Binop_Concatenate {
  public:
    
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


    char const * get_label      () const { return "Binop_Concatenate"; }
    YAKL_INLINE int get_num_inputs () const { return params.num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return params.num_outputs; }


    YAKL_INLINE void compute_one_output(realConst2d input1, realConst2d input2,
                                        real2d const &output, int ibatch, int irow) const {
      if (params.after) {
        int num_inputs_1 = input1.dimension[0];
        output(irow,ibatch) = irow < num_inputs_1 ? input1(irow,ibatch) : input2(irow - num_inputs_1,ibatch);
      } else {
        int num_inputs_2 = input2.dimension[0];
        output(irow,ibatch) = irow < num_inputs_2 ? input2(irow,ibatch) : input1(irow - num_inputs_2,ibatch);
      }
    }


    void print_verbose() const {
      std::cout << "    concatenating saved index, " << index << ", onto the previous layer's output\n";
    }


    void validate() const { }

  };

}


