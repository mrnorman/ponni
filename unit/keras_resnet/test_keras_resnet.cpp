
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  using ponni::load_h5_weights;
  using ponni::Matvec;
  using ponni::Bias;
  using ponni::Silu;
  using ponni::Save_State;
  using ponni::Binop_Add;
  Kokkos::initialize( argc , argv );
  {
    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.h5>" << std::endl;
      return -1;
    }

    // This is the file with the saved tensorflow weights
    std::string fname_h5 = argv[1];

    int   neurons = 20;
    auto model = create_inference_model(
                    // Layer 1
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense/dense"     , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense/dense"     , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 2
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_1/dense_1" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_1/dense_1" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 3
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_2/dense_2" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_2/dense_2" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 4
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_3/dense_3" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_3/dense_3" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 5
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_4/dense_4" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_4/dense_4" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 6
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_5/dense_5" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_5/dense_5" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 7
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_6/dense_6" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_6/dense_6" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0>( neurons )                                                                ,
                    Save_State<0>( neurons )                                                                ,
                    // Layer 8
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_7/dense_7" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_7/dense_7" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 9
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_8/dense_8" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_8/dense_8" , "bias:0"   ) ) ,
                    Silu        <float>( neurons )                                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    // Layer 10
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_9/dense_9" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_9/dense_9" , "bias:0"   ) ) );

    model.validate();
    model.print();
    auto model_as_array = model.represent_as_array();
    model.set_layers_from_array_representation( model_as_array );
    model.save_to_text_file("keras_resnet_save.txt");
    model.load_from_text_file("keras_resnet_save.txt");

    auto &layer = model.get_layer<5>();

    std::cout << "*** TOTAL TRAINABLE PARAMETERS: " << model.get_num_trainable_parameters() << std::endl;

    {
      auto inputs   = ponni::load_h5_weights<2>( fname_h5 , "/test" , "input"  );
      auto expected = ponni::load_h5_weights<2>( fname_h5 , "/test" , "output" );
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
      auto inputs = ponni::load_h5_weights<2>( fname_h5 , "/test" , "input" );
      auto expected = ponni::load_h5_weights<2>( fname_h5 , "/test" , "output" );

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
