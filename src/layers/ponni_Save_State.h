
#pragma once
// Included by ponni.h

namespace ponni {

  template <int N>
  class Save_State {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = true;
    int  static constexpr index           = N;

    struct Params {
      int num_inputs;
      int num_outputs;
    };

    Params params;

    Save_State() {}
    Save_State(int num_inputs) {
      init(num_inputs);
    }


    void init(int num_inputs) {
      params.num_inputs  = num_inputs;
      params.num_outputs = num_inputs;
    }


    char const * get_label         () const { return "Save_State"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE static void compute_all_outputs(real2d const &input, real2d const &output, int ibatch, Params const &params_in) {
      for (int irow = 0; irow < params_in.num_outputs; irow++) {
        output(irow,ibatch) = input(irow,ibatch);
      }
    }


    void print_verbose() const {
      std::cout << "    saving into index: " << index << "\n";
    }


    void validate() const { }

  };

}


