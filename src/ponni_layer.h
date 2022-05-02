
#pragma once

#include "layers/Dense_matmul.h"
#include "layers/Dense_bias.h"
#include "layers/Act_relu.h"

namespace ponni {

  class Layer {
  protected:
    
    realConst1d wts;     // Weights and parameters
    int         type;    // What is the operation being performed by this node?
    int         num_in;  // How many inputs we expect to apply this node on
    int         num_out; // How many outputs we expect to apply this node on
    bool        ovr_wrt; // Whether input vector can be overwritten

    YAKL_INLINE void copy_data(Layer const &rhs) {
      this->wts     = rhs.wts    ;
      this->type    = rhs.type   ;
      this->num_in  = rhs.num_in ;
      this->num_out = rhs.num_out;
      this->ovr_wrt = rhs.ovr_wrt;
    }

    YAKL_INLINE void nullify() {
      this->wts     = realConst1d();
      this->type    = UNINITIALIZED;
      this->num_in  = -1;
      this->num_out = -1;
      this->ovr_wrt = false;
    }

  public:
    
    YAKL_INLINE Layer () { nullify(); }
    YAKL_INLINE ~Layer() { nullify(); }
    YAKL_INLINE Layer                  (Layer const  &rhs) { copy_data(rhs); }
    YAKL_INLINE Layer                  (Layer const &&rhs) { copy_data(rhs); }
    YAKL_INLINE Layer const & operator=(Layer const  &rhs) { if (this != &rhs) copy_data(rhs); return *this; }
    YAKL_INLINE Layer const & operator=(Layer const &&rhs) { if (this != &rhs) copy_data(rhs); return *this; }


    // Accessors
    realConst1d      get_weights        () const { return this->wts    ; }
    YAKL_INLINE int  get_num_inputs     () const { return this->num_in ; }
    YAKL_INLINE int  get_num_outputs    () const { return this->num_out; }
    YAKL_INLINE int  get_type           () const { return this->type   ; }
    YAKL_INLINE bool get_overwrite_input() const { return this->ovr_wrt; }
    std::string      get_type_str() const {
      if      (type == TYPE_DENSE_MATMUL  ) { return "Dense Matmul"   ; }
      else if (type == TYPE_DENSE_ADD_BIAS) { return "Dense Add Bias" ; }
      else if (type == TYPE_ACT_RELU      ) { return "Activaiton ReLU"; }
      return "";
    }


    // Apply this layer by parallelizing only over batches, not over rows
    YAKL_INLINE void apply_batch_parallel( realConst2d input , real2d const &output , int ibatch ) const {
      if (type == TYPE_DENSE_MATMUL  ) {
        for (int irow=0; irow<num_out; irow++) { Dense_matmul::apply_1(wts,num_in,num_out,input,output,ibatch,irow); }
        return;
      }
      if (type == TYPE_DENSE_ADD_BIAS) {
        for (int irow=0; irow<num_out; irow++) { Dense_bias  ::apply_1(wts,num_in,num_out,input,output,ibatch,irow); }
        return;
      }
      if (type == TYPE_ACT_RELU      ) {
        for (int irow=0; irow<num_out; irow++) { Act_relu    ::apply_1(wts,num_in,num_out,input,output,ibatch,irow); }
        return;
      }
    }


    void init( int type , int num_in , int num_out , realConst1d wts = realConst1d() ) {
      this->type    = type   ;
      this->num_in  = num_in ;
      this->num_out = num_out;
      this->wts     = wts    ;
      if      (type == TYPE_DENSE_MATMUL  ) { Dense_matmul::init(num_in,num_out,wts,ovr_wrt); }
      else if (type == TYPE_DENSE_ADD_BIAS) { Dense_bias  ::init(num_in,num_out,wts,ovr_wrt); }
      else if (type == TYPE_ACT_RELU      ) { Act_relu    ::init(num_in,num_out,wts,ovr_wrt); }
    }


    YAKL_INLINE void apply_1( realConst2d input , real2d const &output , int ibatch , int irow ) const {
    }


    void print_verbose() const {
      std::cout << std::setw(15) << std::left << get_type_str() << " with "
                << num_in  << " inputs and "
                << num_out << " outputs.\n";
      if      (type == TYPE_DENSE_MATMUL  ) { Dense_matmul::print_verbose(wts,num_in,num_out); }
      else if (type == TYPE_DENSE_ADD_BIAS) { Dense_bias  ::print_verbose(wts,num_in,num_out); }
      else if (type == TYPE_ACT_RELU      ) { Act_relu    ::print_verbose(wts,num_in,num_out); }
    }

  };

}


