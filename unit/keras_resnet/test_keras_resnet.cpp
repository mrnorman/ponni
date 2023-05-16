
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  using ponni::load_h5_weights;
  using ponni::Matvec;
  using ponni::Bias;
  using ponni::Relu;
  using ponni::Save_State;
  using ponni::Binop_Add;
  yakl::init();
  {
    // This is the file with the saved tensorflow weights
    std::string fname_h5 = argv[1];

    int   neurons = 20;
    float negative_slope = 0.1;

    auto model = create_inference_model(
                    // Layer 1
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense/dense"     , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense/dense"     , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 2
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_1/dense_1" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_1/dense_1" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 3
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_2/dense_2" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_2/dense_2" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 4
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_3/dense_3" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_3/dense_3" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 5
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_4/dense_4" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_4/dense_4" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 6
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_5/dense_5" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_5/dense_5" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 7
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_6/dense_6" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_6/dense_6" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0>( neurons )                                                          ,
                    Save_State<0>( neurons )                                                          ,
                    // Layer 8
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_7/dense_7" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_7/dense_7" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
                    Binop_Add <0,float>( neurons )                                                          ,
                    Save_State<0,float>( neurons )                                                          ,
                    // Layer 9
                    Matvec      <float>( load_h5_weights<2>( fname_h5 , "/dense_8/dense_8" , "kernel:0" ) ) ,
                    Bias        <float>( load_h5_weights<1>( fname_h5 , "/dense_8/dense_8" , "bias:0"   ) ) ,
                    Relu        <float>( neurons , negative_slope )                                         ,
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
      // Load one test sample to ensure we're getting the same outputs
      std::array<float,137> datavec = {0.49597812,0.49602726,0.49610856,0.49575031,0.49577570,0.49590355,
                                       0.49576440,0.49588415,0.49586010,0.49584466,0.49595755,0.49603355,
                                       0.49559316,0.49563473,0.49568036,0.49574825,0.49578935,0.49574974,
                                       0.49557942,0.49576527,0.49591160,0.49547020,0.49564469,0.49570984,
                                       0.49553356,0.49575782,0.49577305,0.45877230,0.45875913,0.45876855,
                                       0.45882720,0.45878604,0.45879951,0.45883587,0.45881239,0.45883536,
                                       0.45878452,0.45883051,0.45879772,0.45889884,0.45889127,0.45885456,
                                       0.45886090,0.45889524,0.45883015,0.45880255,0.45884499,0.45881921,
                                       0.45893389,0.45892829,0.45890638,0.45890066,0.45890975,0.45888293,
                                       0.48934004,0.48931572,0.48930022,0.48938042,0.48936340,0.48934978,
                                       0.48942113,0.48940295,0.48939061,0.48931345,0.48931283,0.48932874,
                                       0.48935172,0.48932892,0.48933974,0.48937979,0.48934701,0.48936093,
                                       0.48933807,0.48933282,0.48934740,0.48938069,0.48936185,0.48934007,
                                       0.48940080,0.48935324,0.48932013,0.52962625,0.52961826,0.52962226,
                                       0.52959174,0.52960217,0.52961057,0.52957541,0.52956986,0.52959603,
                                       0.52959400,0.52961093,0.52963167,0.52957004,0.52957320,0.52962196,
                                       0.52960086,0.52957922,0.52956444,0.52956223,0.52961594,0.52965915,
                                       0.52954853,0.52959788,0.52963775,0.52955705,0.52961016,0.52961469,
                                       0.74753761,0.74752748,0.74755841,0.74758607,0.74755615,0.74751335,
                                       0.74755186,0.74756181,0.74751490,0.74753714,0.74754232,0.74750519,
                                       0.74753839,0.74756271,0.74750465,0.74751884,0.74753231,0.74750185,
                                       0.74752432,0.74751109,0.74751288,0.74754196,0.74753088,0.74754441,
                                       0.74752563,0.74750173,0.74751961,0.51605195,0.44940680};
      yakl::Array<float,2,yakl::memHost> inputs("inputs",datavec.data(),137,1);

      auto out_host = model.forward_batch_parallel( inputs.createDeviceCopy() ).createHostCopy();

      std::cout << "Absolute difference for Output 1: " << std::abs( out_host(0,0) - 0.50536489 ) << std::endl;
      std::cout << "Absolute difference for Output 2: " << std::abs( out_host(1,0) - 0.54642177 ) << std::endl;
      std::cout << "Absolute difference for Output 3: " << std::abs( out_host(2,0) - 0.54819602 ) << std::endl;
      std::cout << "Absolute difference for Output 4: " << std::abs( out_host(3,0) - 0.43084893 ) << std::endl;
      std::cout << "Absolute difference for Output 5: " << std::abs( out_host(4,0) - 0.52051890 ) << std::endl;

      if ( std::abs( out_host(0,0) - 0.50536489 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 1 diff too large");
      if ( std::abs( out_host(1,0) - 0.54642177 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 2 diff too large");
      if ( std::abs( out_host(2,0) - 0.54819602 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 3 diff too large");
      if ( std::abs( out_host(3,0) - 0.43084893 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 4 diff too large");
      if ( std::abs( out_host(4,0) - 0.52051890 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 5 diff too large");
    }


    {
      // Load one test sample to ensure we're getting the same outputs
      std::array<float,137> datavec = {0.49597812,0.49602726,0.49610856,0.49575031,0.49577570,0.49590355,
                                       0.49576440,0.49588415,0.49586010,0.49584466,0.49595755,0.49603355,
                                       0.49559316,0.49563473,0.49568036,0.49574825,0.49578935,0.49574974,
                                       0.49557942,0.49576527,0.49591160,0.49547020,0.49564469,0.49570984,
                                       0.49553356,0.49575782,0.49577305,0.45877230,0.45875913,0.45876855,
                                       0.45882720,0.45878604,0.45879951,0.45883587,0.45881239,0.45883536,
                                       0.45878452,0.45883051,0.45879772,0.45889884,0.45889127,0.45885456,
                                       0.45886090,0.45889524,0.45883015,0.45880255,0.45884499,0.45881921,
                                       0.45893389,0.45892829,0.45890638,0.45890066,0.45890975,0.45888293,
                                       0.48934004,0.48931572,0.48930022,0.48938042,0.48936340,0.48934978,
                                       0.48942113,0.48940295,0.48939061,0.48931345,0.48931283,0.48932874,
                                       0.48935172,0.48932892,0.48933974,0.48937979,0.48934701,0.48936093,
                                       0.48933807,0.48933282,0.48934740,0.48938069,0.48936185,0.48934007,
                                       0.48940080,0.48935324,0.48932013,0.52962625,0.52961826,0.52962226,
                                       0.52959174,0.52960217,0.52961057,0.52957541,0.52956986,0.52959603,
                                       0.52959400,0.52961093,0.52963167,0.52957004,0.52957320,0.52962196,
                                       0.52960086,0.52957922,0.52956444,0.52956223,0.52961594,0.52965915,
                                       0.52954853,0.52959788,0.52963775,0.52955705,0.52961016,0.52961469,
                                       0.74753761,0.74752748,0.74755841,0.74758607,0.74755615,0.74751335,
                                       0.74755186,0.74756181,0.74751490,0.74753714,0.74754232,0.74750519,
                                       0.74753839,0.74756271,0.74750465,0.74751884,0.74753231,0.74750185,
                                       0.74752432,0.74751109,0.74751288,0.74754196,0.74753088,0.74754441,
                                       0.74752563,0.74750173,0.74751961,0.51605195,0.44940680};
      yakl::Array<float,3,yakl::memHost> inputs_host("inputs",datavec.data(),137,1,1);
      auto inputs = inputs_host.createDeviceCopy();

      model.init( 1 , 1 );
      yakl::Array<float,3,yakl::memDevice> outputs("outputs",5,1,1);
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<2>(1,1) ,
                                                 YAKL_LAMBDA (int ibatch, int iens) {
        model.forward_in_kernel( inputs , outputs , model.params , ibatch , iens );
      });
      auto out_host = outputs.createHostCopy();

      std::cout << "Absolute difference for Output 1: " << std::abs( out_host(0,0,0) - 0.50536489 ) << std::endl;
      std::cout << "Absolute difference for Output 2: " << std::abs( out_host(1,0,0) - 0.54642177 ) << std::endl;
      std::cout << "Absolute difference for Output 3: " << std::abs( out_host(2,0,0) - 0.54819602 ) << std::endl;
      std::cout << "Absolute difference for Output 4: " << std::abs( out_host(3,0,0) - 0.43084893 ) << std::endl;
      std::cout << "Absolute difference for Output 5: " << std::abs( out_host(4,0,0) - 0.52051890 ) << std::endl;

      if ( std::abs( out_host(0,0,0) - 0.50536489 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 1 diff too large");
      if ( std::abs( out_host(1,0,0) - 0.54642177 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 2 diff too large");
      if ( std::abs( out_host(2,0,0) - 0.54819602 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 3 diff too large");
      if ( std::abs( out_host(3,0,0) - 0.43084893 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 4 diff too large");
      if ( std::abs( out_host(4,0,0) - 0.52051890 ) > 1.e-6 ) yakl::yakl_throw("ERROR Output 5 diff too large");
    }

  }
  yakl::finalize();
}

