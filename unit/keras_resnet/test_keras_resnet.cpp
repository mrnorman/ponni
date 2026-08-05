
#include "ponni.h"

int main( int argc , char **argv ) {
  using ponni::load_ponni_tensor;
  using ponni::Matvec;
  using ponni::Bias;
  using ponni::Silu;
  using ponni::Save_State;
  using ponni::Binop_Add;
  Kokkos::initialize( argc , argv );
  {
    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.ponni>" << std::endl;
      return -1;
    }

    std::string fname = argv[1];
    ponni::PonniFile file;
    std::string error;
    if (!file.load(fname,&error)) throw std::runtime_error(error);

    int   neurons = 20;
    auto model = create_inference_model(
                    // Layer 1
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 2
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_1.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_1.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 3
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_2.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_2.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 4
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_3.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_3.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 5
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_4.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_4.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 6
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_5.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_5.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 7
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_6.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_6.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0>( neurons )                                                                ,
                    Save_State<0>( neurons )                                                                ,
                    // Layer 8
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_7.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_7.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 9
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_8.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_8.bias") ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    // Layer 10
                    Matvec      <float>( load_ponni_tensor<2>(file,"dense_9.kernel") ) ,
                    Bias        <float>( load_ponni_tensor<1>(file,"dense_9.bias") ) );

    model.validate();
    model.print();

    auto &layer = model.get_layer<5>();

    std::cout << "*** TOTAL TRAINABLE PARAMETERS: " << model.get_num_trainable_parameters() << std::endl;

    {
      auto inputs   = load_ponni_tensor<2>(file,"test.input");
      auto expected = load_ponni_tensor<2>(file,"test.output");
      auto outputs  = model.forward_batch_parallel( inputs );

      auto out_host = ponni::create_host_copy(outputs);
      auto exp_host = ponni::create_host_copy(expected);

      if (out_host.extent(0) != exp_host.extent(0) || out_host.extent(1) != exp_host.extent(1)) {
        Kokkos::abort("ERROR: output dimensions do not match expected dimensions");
      }

      for (int j = 0; j < out_host.extent(1); j++) {
        for (int i = 0; i < out_host.extent(0); i++) {
          float diff = std::abs(out_host(i,j) - exp_host(i,j));
          std::cout << "Absolute difference for Output(" << i << "," << j << "): " << diff << std::endl;
          if (diff > 1.e-5f) Kokkos::abort("ERROR: output diff too large");
        }
      }
    }


    {
      auto inputs = load_ponni_tensor<2>(file,"test.input");
      auto expected = load_ponni_tensor<2>(file,"test.output");

      model.reallocate_internal_state( inputs.extent(1) );
      Kokkos::View<float**,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space> outputs("outputs",expected.extent(0),expected.extent(1));
      Kokkos::parallel_for( PONNI_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int ibatch) {
        model.forward_batch_parallel_in_kernel( inputs , outputs , model.params , ibatch );
      });
      auto out_host = ponni::create_host_copy( outputs );

      auto exp_host = ponni::create_host_copy(expected);
      if (out_host.extent(0) != exp_host.extent(0) || out_host.extent(1) != exp_host.extent(1)) {
        Kokkos::abort("ERROR: output dimensions do not match expected dimensions");
      }

      for (int j = 0; j < out_host.extent(1); j++) {
        for (int i = 0; i < out_host.extent(0); i++) {
          float diff = std::abs(out_host(i,j) - exp_host(i,j));
          std::cout << "Absolute difference for Output(" << i << "," << j << "): " << diff << std::endl;
          if (diff > 1.e-5f) Kokkos::abort("ERROR: output diff too large");
        }
      }
    }

  }
  Kokkos::finalize();
}
