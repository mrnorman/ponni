# PONNI: Portable Online Neural Network Inferencing
### Efficient in-loop neural network inferencing made easy in C++

Author: Matt Norman, Oak Ridge National Laboratory, https://mrnorman.github.io

PONNI provides a convenient way to build an efficient, portable Neural Network inference model in C++ with minimal syntax and full disclosure of exactly how the model is running on an accelerator device. It is built on the [Yet Another Kernel Launcher (YAKL)](https://github.com/mrnorman/YAKL) portable C++ library and runs out of the box on Nvidia, AMD, and Intel GPUs as well as CPUs with or without OpenMP 3.5 threading.

PONNI currently supports:
* Dense matrix multiply and bias addition layers
* A variety of activation function layers
* The ability to concatenate or add two vectors of neurons for DenseNet and ResNet architectures, respectively

An example of creating a combined DenseNet and ResNet is given below. This isn't necessarily an architecture you would find useful, but it demonstrates how to use ponni. In the example below, the kernel generated for `model.batch_parallel()` for an Nvidia sm_61 architecture only used 32 registers, meaning it will efficiently occupy the device.

```C++
#include "ponni.h"

#include "ponni.h"
#include "ponni_load_tensorflow_h5_weights.h"

int main( int argc , char **argv ) { 
  yakl::init();
  {
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;
    using ponni::Binop_Concatenate;
    using ponni::real1d;  // All "real1d" and "real2d" arrays are in GPU device memory, not host memory
    using ponni::real2d;
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    Matvec               layer0 ( real2d("matrix_1",12,10) );
    Bias                 layer1 ( real1d("bias_1",10) )     ;   
    Relu                 layer2 ( 10 , 0.1 )                ;
    Save_State<0>        layer3 ( 10 )                      ; // Save output of layer2
                                                              //   into saved index 0.
    Matvec               layer4 ( real2d("matrix_2",10,10) );
    Bias                 layer5 ( real1d("bias_2",10) )     ;   
    Relu                 layer6 ( 10 , 0.1 )                ;
    Binop_Add<0>         layer7 ( 10 )                      ; // Output of layer2 added
                                                              //   to output of layer6.
    Matvec               layer8 ( real2d("matrix_2",10,20) );
    Bias                 layer9 ( real1d("bias_2",20) )     ;   
    Relu                 layer10( 20 , 0.1 )                ;
    Save_State<0>        layer11( 20 )                      ; // Save output of layer11
                                                              //   into saved index 0.
                                                              // Reusing indices reduces
                                                              //   memory usage.
    Matvec               layer12( real2d("matrix_3",20,8) ) ; 
    Bias                 layer13( real1d("bias_3",8) )      ;   
    Relu                 layer14( 8 , 0.1 )                 ;
    Binop_Concatenate<0> layer15( 8 , 28 )                  ; // Output of layer11
                                                              //   concatenated after
                                                              //   output of layer 14.
    Matvec               layer16( real2d("matrix_4",28,4) ) ; 
    Bias                 layer17( real1d("bias_4",4) )      ;   

    // Load weights into layers 0, 1, 4, 5, 8, 9, 12, 13, 16, 17 from a file
    // Matrices must be in column,row format; not in row,column format
    // This is the default ordering for Keras saved weights hdf5 files
    // You must pay attention to the ordering for the weights you save from other libraries

    // Create an inference model to perform batched forward predictions
    // You must place the layers yourself in this fashion. PONNI wraps these in std::tuple
    //   to create an efficient model resolved at compile time with templates.
    auto model = create_inference_model( layer0  , layer1  , layer2  , layer3  , layer4  ,
                                         layer5  , layer6  , layer7  , layer8  , layer9  ,
                                         layer10 , layer11 , layer12 , layer13 , layer14 ,
                                         layer15 , layer16 , layer17 );
                                                   
    model.print()          // Prints basic information about each layer to stdout
    model.print_verbose(); // Prints detailed information about each layer to stdout

    // Create arrays for inputs 
    int batch_size = [number of inputs you want to inference];
    real2d inputs ("inputs",12,num_batches);
    
    // Populate inputs on the GPU, or populate inputs on the host and transfer to the GPU
    
    // Run the model, parallelizing the batch index only on the GPU.
    // The entire model runs in a single GPU kernel.
    // The output real2d array is created, allocated, and populated for you.
    // It exists in GPU device memory.
    real2d outputs = model.batch_parallel( inputs );
    
    // Do whatever you want with the model outputs here
  }
  yakl::finalize();
}

