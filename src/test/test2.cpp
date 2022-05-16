
#include "ponni.h"
#include "ponni_load_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;

    std::string fname = "supercell_micro_PyTorch_ResNet_1000000_Nneu5_modelparameters.h5";

    // Create layers & load weights
    bool transpose = true;
    ponni::Matvec matvec_1( ponni::load_h5_weights<2>( fname , "/" , "0.0.0.0.1.weight"            , transpose ) );
    ponni::Bias   bias_1  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.0.0.1.bias"              , transpose ) );
    ponni::Matvec matvec_2( ponni::load_h5_weights<2>( fname , "/" , "0.0.0.2.sequential.0.weight" , transpose ) );
    ponni::Bias   bias_2  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.0.2.sequential.0.bias"   , transpose ) );
    ponni::Matvec matvec_3( ponni::load_h5_weights<2>( fname , "/" , "0.0.2.sequential.0.weight"   , transpose ) );
    ponni::Bias   bias_3  ( ponni::load_h5_weights<1>( fname , "/" , "0.0.2.sequential.0.bias"     , transpose ) );
    ponni::Matvec matvec_4( ponni::load_h5_weights<2>( fname , "/" , "0.2.sequential.0.weight"     , transpose ) );
    ponni::Bias   bias_4  ( ponni::load_h5_weights<1>( fname , "/" , "0.2.sequential.0.bias"       , transpose ) );
    ponni::Matvec matvec_5( ponni::load_h5_weights<2>( fname , "/" , "2.weight"                    , transpose ) );
    ponni::Bias   bias_5  ( ponni::load_h5_weights<1>( fname , "/" , "2.bias"                      , transpose ) );

    // Create an inference model to perform batched forward predictions
    auto inference = create_inference_model( matvec_1           ,
                                             bias_1             ,
                                             Relu( 5 , 0.1 )    ,
                                             Save_State<0>( 5 ) ,
                                             matvec_2           ,
                                             bias_2             ,
                                             Relu( 5 , 0.1 )    ,
                                             Binop_Add<0>( 5 )  ,
                                             Save_State<0>( 5 ) ,
                                             matvec_3           ,
                                             bias_3             ,
                                             Relu( 5 , 0.1 )    ,
                                             Binop_Add<0>( 5 )  ,
                                             Save_State<0>( 5 ) ,
                                             matvec_4           ,
                                             bias_4             ,
                                             Relu( 5 , 0.1 )    ,
                                             Binop_Add<0>( 5 )  ,
                                             matvec_5           ,
                                             bias_5             );
                                                   
    inference.validate();
    inference.print_verbose();

    // yakl::Array<float,2,yakl::memHost,yakl::styleC> inputs("inputs",12,1);
    // inputs( 0,0) = 5.08810276e-01;
    // inputs( 1,0) = 4.78929254e-01;
    // inputs( 2,0) = 4.54260898e-01;
    // inputs( 3,0) = 6.02555739e-02;
    // inputs( 4,0) = 4.85583159e-02;
    // inputs( 5,0) = 3.79940443e-02;
    // inputs( 6,0) = 1.20564349e-04;
    // inputs( 7,0) = 6.27402543e-04;
    // inputs( 8,0) = 3.41872996e-03;
    // inputs( 9,0) = 1.34502158e-03;
    // inputs(10,0) = 4.29940776e-04;
    // inputs(11,0) = 6.08758314e-06;

    // // Perform a batched inference
    // auto outputs = inference.batch_parallel( inputs.createDeviceCopy() );
  }
  yakl::finalize();
}

