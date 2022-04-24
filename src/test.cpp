
#include "pinni.h"
#include "pinni_load_keras.h"

int main( int argc , char **argv ) {
  yakl::init();
  {
    auto model = pinni::load_keras_model<3>( argv[1] , argv[2] );
    model.print_verbose();
  }
  yakl::finalize();
}

