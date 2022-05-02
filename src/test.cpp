
#include "ponni.h"
#include "ponni_load_keras.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    auto model = ponni::load_keras_model<10>( argv[1] , argv[2] );
    model.print_verbose();

    yakl::Array<float,2,yakl::memHost,yakl::styleC> inputs("inputs",12,1);
    inputs( 0,0) = 0.5088102764679303113837250f;
    inputs( 1,0) = 0.4789292535713394194374359f;
    inputs( 2,0) = 0.4542608979565114779575197f;
    inputs( 3,0) = 0.0602555739083469252270753f;
    inputs( 4,0) = 0.0485583158794242186750978f;
    inputs( 5,0) = 0.0379940443092740554043019f;
    inputs( 6,0) = 0.0001205643487987410764290f;
    inputs( 7,0) = 0.0006274025431650152384924f;
    inputs( 8,0) = 0.0034187299600010700832697f;
    inputs( 9,0) = 0.0013450215768994594738722f;
    inputs(10,0) = 0.0004299407756233863931415f;
    inputs(11,0) = 0.0000060875831392092127889f;

    auto outputs = model.inference_batchparallel( inputs.createDeviceCopy() );

    std::cout << outputs;

    // 4.7603926e-01 4.7099892e-02 3.0458346e-04 1.3100356e-03
  }
  yakl::finalize();
}

