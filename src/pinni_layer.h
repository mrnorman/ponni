
#pragma once

namespace pinni {

  class Layer {
  protected:
    
    real2d kernel;      // Matrix kernel for dense layers in column,row format
    real1d bias;        // Bias vector for dense layers
    real1d params;      // Miscellaneous parameters for things like activation functions
    int    type;        // What is the operation being performed by this node?
    int    num_inputs;  // How many inputs we expect to apply this node on
    int    num_outputs; // How many outputs we expect to apply this node on

    YAKL_INLINE void copy_data(Layer const &rhs) {
      this->kernel      = rhs.kernel     ;
      this->bias        = rhs.bias       ;
      this->params      = rhs.params     ;
      this->type        = rhs.type       ;
      this->num_inputs  = rhs.num_inputs ;
      this->num_outputs = rhs.num_outputs;
    }

  public:
    
    YAKL_INLINE Layer() {
      this->type        = UNINITIALIZED;
      this->num_inputs  = -1;
      this->num_outputs = -1;
    }
    YAKL_INLINE ~Layer() {
      this->kernel      = real2d();
      this->bias        = real1d();
      this->params      = real1d();
      this->type        = UNINITIALIZED;
      this->num_inputs  = -1;
      this->num_outputs = -1;
    }
    YAKL_INLINE Layer                  (Layer const  &rhs) { copy_data(rhs); }
    YAKL_INLINE Layer                  (Layer const &&rhs) { copy_data(rhs); }
    YAKL_INLINE Layer const & operator=(Layer const  &rhs) { if (this == &rhs) return *this; copy_data(rhs); return *this; }
    YAKL_INLINE Layer const & operator=(Layer const &&rhs) { if (this == &rhs) return *this; copy_data(rhs); return *this; }


    // Modifiers
    void set_kernel( real2d const &kernel ) {
      this->kernel      = kernel;
      this->num_inputs  = kernel.dimension[0];
      this->num_outputs = kernel.dimension[1];
    }
    void set_bias( real1d const &bias ) {
      this->bias = bias;
      this->num_outputs = bias.dimension[0];
    }
    void set_num_inputs( int num_inputs ) {
      this->num_inputs  = num_inputs;
    }
    void set_num_outputs( int num_outputs ) {
      this->num_outputs  = num_outputs;
    }
    void set_params( real1d const &params ) {
      this->params = params;
    }
    void set_type( int type ) {
      this->type = type;
    }


    // Accessors
    real2d get_kernel() const { return this->kernel     ; }
    real1d get_bias  () const { return this->bias       ; }
    real1d get_params() const { return this->params     ; }
    YAKL_INLINE int get_num_inputs () const { return this->num_inputs ; }
    YAKL_INLINE int get_num_outputs() const { return this->num_outputs; }
    YAKL_INLINE int get_type       () const { return this->type       ; }
    std::string get_type_str() const {
      if      (type == TYPE_DENSE                ) { return "Dense"    ; }
      else if (type == TYPE_ACTIVATION_RELU      ) { return "ReLU"     ; }
      else if (type == TYPE_ACTIVATION_LEAKY_RELU) { return "LeakyReLU"; }
      return "";
    }


    // Apply this layer serially over the inputs for this batch index
    // For serial applications, the "row / num_output" index is first, and the batch index is last
    YAKL_INLINE void apply_serial( realConst2d input , real2d const &output , int ibatch ) const {
      for (int irow = 0; irow < output.dimension[0]; irow++) {
        apply_1( input , output , ibatch , irow );
      }
    }


    std::string validate() const {
      if (type == UNINITIALIZED) return "Node not initialized";
      if (num_inputs  <= 0) return "Node's num_inputs not initialized";
      if (num_outputs <= 0) return "Node's num_outputs not initialized";
      if      (type == TYPE_DENSE                ) { return validate_dense                (); }
      else if (type == TYPE_ACTIVATION_RELU      ) { return validate_activation_relu      (); }
      else if (type == TYPE_ACTIVATION_LEAKY_RELU) { return validate_activation_leaky_relu(); }
      else                                         { return "Invalid node type"; }
      return "";
    }


    YAKL_INLINE void apply_1( realConst2d input , real2d const &output , int ibatch , int irow ) const {
      // This is purposefully structured this way becuase eventually, we'll have a templated bitmask
      // to determine which layers to enable and which to disable. With if constexpr, we'll be able to
      // allow the compiler to exclude code we don't need for skinnier kerenls.
      if (this->type == TYPE_DENSE) {
        apply_dense_1( input , output , ibatch , irow );
        return;
      }
      if (this->type == TYPE_ACTIVATION_RELU) {
        apply_activation_relu_1( input , output , ibatch , irow );
        return;
      }
      if (this->type == TYPE_ACTIVATION_LEAKY_RELU) {
        apply_activation_leaky_relu_1( input , output , ibatch , irow );
        return;
      }
    }


    // Dense
    std::string validate_dense() const {
      if ( ! kernel.initialized() ) return "Dense Node kernel matrix not initialized";
      if ( ! bias.initialized() ) return "Dense Node bias vector not initialized";
      if ( kernel.dimension[0] != num_inputs) return "Dense Node kernel matrix # columns != num_inputs";
      if ( kernel.dimension[1] != num_outputs) return "Dense Node kernel matrix # rows != num_outputs";
      if ( bias.dimension[0] != num_outputs) return "Dense Node kernel bias # rows != num_outputs";
      return "";
    }
    YAKL_INLINE void apply_dense_1( realConst2d input , real2d const &output , int ibatch , int irow ) const {
      real tmp = 0;
      for (int k=0; k < input.dimension[0]; k++) {
        tmp += kernel(k,irow) * input(k,ibatch);
      }
      output(irow,ibatch) = tmp + bias(irow);
    }
    void print_verbose_dense() const {
      std::cout << "    kernel:\n";
      auto kernel_host = kernel.createHostCopy();
      for (int irow=0; irow < num_outputs; irow++) {
        std::cout << "      ";
        for (int icol=0; icol < num_inputs; icol++) {
          std::cout << std::setw(12) << kernel_host(icol,irow) << "  ";
        }
        std::cout << "\n";
      }
      std::cout << "    bias:\n";
      auto bias_host = bias.createHostCopy();
      for (int irow=0; irow < num_outputs; irow++) {
        std::cout << "      " << std::setw(12) << bias_host(irow) << "\n";
      }
    }


    // ReLU
    std::string validate_activation_relu() const {
      if ( num_inputs != num_outputs ) return "ReLU Node inputs != outputs";
      return "";
    }
    YAKL_INLINE void apply_activation_relu_1( realConst2d input , real2d const &output , int ibatch , int irow ) const {
      output(irow,ibatch) = std::max( input(irow,ibatch) , static_cast<real>(0.) );
    }
    void print_verbose_activation_relu() const {}


    // LeakyReLU
    std::string validate_activation_leaky_relu() const {
      if ( num_inputs != num_outputs ) return "LeakyReLU Node inputs != outputs";
      if ( ! params.initialized() ) return "LeakyReLU Node params not initialized";
      if ( params.dimension[0] != 1 ) return "LeakyReLU Node params has more than one element";
      return "";
    }
    YAKL_INLINE void apply_activation_leaky_relu_1( realConst2d input , real2d const &output , int ibatch , int irow ) const {
      output(irow,ibatch) = input(irow,ibatch) >= 0 ? input(irow,ibatch) : params(0) * input(irow,ibatch);
    }
    void print_verbose_activation_leaky_relu() const {
      std::cout << "    slope: " << std::setw(12) << params.createHostCopy()(0) << "\n";
    }


    void print_verbose() const {
      std::cout << std::setw(15) << std::left << get_type_str() << " with "
                << num_inputs << " inputs and "
                << num_outputs << " outputs.\n";
      if      (type == TYPE_DENSE                ) { print_verbose_dense                (); }
      else if (type == TYPE_ACTIVATION_RELU      ) { print_verbose_activation_relu      (); }
      else if (type == TYPE_ACTIVATION_LEAKY_RELU) { print_verbose_activation_leaky_relu(); }
    }

  };

}


