
#include "ponni.h"
#include "ponni_load_tensorflow_h5_weights.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    std::string fname = "supercell_micro_Keras_modelwt_NORMip_NORMop1000000_Nneu10.h5";

    ponni::real2d matvec_1_weights("matvec_1_weights",12,10);
    ponni::load_tensorflow_h5_weights( matvec_1_weights , fname , "/dense/dense" , "kernel:0" );
    ponni::Matvec matvec_1( matvec_1_weights );

    ponni::real1d bias_1_weights("bias_1_weights",10);
    ponni::load_tensorflow_h5_weights( bias_1_weights , fname , "/dense/dense" , "bias:0" );
    ponni::Bias bias_1( bias_1_weights );

    ponni::Relu relu_1( 10, 0.1 );

    ponni::real2d matvec_2_weights("matvec_2_weights",10,4);
    ponni::load_tensorflow_h5_weights( matvec_2_weights , fname , "/dense_1/dense_1" , "kernel:0" );
    ponni::Matvec matvec_2( matvec_2_weights );

    ponni::real1d bias_2_weights("bias_2_weights",4);
    ponni::load_tensorflow_h5_weights( bias_2_weights , fname , "/dense_1/dense_1" , "bias:0" );
    ponni::Bias bias_2( bias_2_weights );

    auto model = ponni::create_model( matvec_1 , bias_1 , relu_1 , matvec_2 , bias_2 );
    model.print_verbose();

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

    auto outputs = model.inference_batch_parallel( inputs.createDeviceCopy() );

    std::cout << outputs;

    // 4.7658795e-01 4.8446856e-02 1.2472458e-03 4.0419400e-05

  }
  yakl::finalize();
}

