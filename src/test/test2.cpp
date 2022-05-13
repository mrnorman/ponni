
#include "ponni.h"
#include "ponni_load_tensorflow_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    // This is the file with the saved tensorflow weights
    std::string fname = "supercell_micro_Keras_modelwt_NORMip_NORMop1000000_Nneu10.h5";

    // Create an inference model to perform batched forward predictions
    auto inference = ponni::create_inference_model( ponni::Matvec              ( ponni::real2d("matrix_1",12,10) ) ,
                                                    ponni::Bias                ( ponni::real1d("bias_1",10) )      ,
                                                    ponni::Relu                ( 10 , 0.1 )                        ,
                                                    ponni::Save_State<0>       ( 10 )                              ,
                                                    ponni::Matvec              ( ponni::real2d("matrix_2",10,10) ) ,
                                                    ponni::Bias                ( ponni::real1d("bias_2",10) )      ,
                                                    ponni::Relu                ( 10 , 0.1 )                        ,
                                                    ponni::Binop_Add<0>        ( 10 )                              ,
                                                    ponni::Save_State<1>       ( 10 )                              ,
                                                    ponni::Matvec              ( ponni::real2d("matrix_3",10,8) )  ,
                                                    ponni::Bias                ( ponni::real1d("bias_3",8) )       ,
                                                    ponni::Relu                ( 8 , 0.1 )                         ,
                                                    ponni::Binop_Concatenate<1>( 8 , 18 )                          ,
                                                    ponni::Matvec              ( ponni::real2d("matrix_4",18,4) )  ,
                                                    ponni::Bias                ( ponni::real1d("bias_4",4) )       );
                                                   
    inference.print_verbose();

    yakl::Array<float,2,yakl::memHost,yakl::styleC> inputs("inputs",12,1);
    inputs( 0,0) = 5.08810276e-01;
    inputs( 1,0) = 4.78929254e-01;
    inputs( 2,0) = 4.54260898e-01;
    inputs( 3,0) = 6.02555739e-02;
    inputs( 4,0) = 4.85583159e-02;
    inputs( 5,0) = 3.79940443e-02;
    inputs( 6,0) = 1.20564349e-04;
    inputs( 7,0) = 6.27402543e-04;
    inputs( 8,0) = 3.41872996e-03;
    inputs( 9,0) = 1.34502158e-03;
    inputs(10,0) = 4.29940776e-04;
    inputs(11,0) = 6.08758314e-06;

    // Perform a batched inference
    auto outputs = inference.batch_parallel( inputs.createDeviceCopy() );

    auto out_host = outputs.createHostCopy();
  }
  yakl::finalize();
}

