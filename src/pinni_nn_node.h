
#pragma once

namespace pinni {

  int constexpr PINNI_UNINITIALIZED              = -1;
  int constexpr PINNI_TYPE_DENSE                 =  1;
  int constexpr PINNI_TYPE_ACTIVATION_RELU       =  2;
  int constexpr PINNI_TYPE_ACTIVATION_LEAKY_RELU =  3;

  class NN_Node {
  protected:
    real2d kernel;
    real1d bias;
    real1d activation_params;
    int    type;

  public:
    
    NN_Node() { this->type = PINNI_UNINITIALIZED; }

    NN_Node(int type) { this->type = type; }

    NN_Node(NN_Node const &rhs) { copy_data(rhs) };

  };

}


