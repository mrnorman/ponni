#include "ponni.h"

#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  bool ok = true;

  {
    if (argc == 1) {
      std::cerr << "Usage: " << argv[0] << " <weights.ponni>" << std::endl;
      return -1;
    }

    std::string fname = argv[1];
    ponni::PonniFile file;
    std::string error;
    if (!file.load(fname,&error)) throw std::runtime_error(error);

    auto w_main = ponni::load_ponni_tensor<2>(file,"w_main");
    auto b_main = ponni::load_ponni_tensor<1>(file,"b_main");
    auto w_skip = ponni::load_ponni_tensor<2>(file,"w_skip");
    auto b_skip = ponni::load_ponni_tensor<1>(file,"b_skip");

    auto model = ponni::create_inference_model(
      ponni::Relu<float>(2),
      ponni::Save_State<0,float>(2),
      ponni::Matvec<float>(w_main, false),
      ponni::Bias<float>(b_main, false),
      ponni::Binop_Projection_Add<0, float, 3, 2>(w_skip, b_skip, false)
    );

    model.validate();

    auto in = ponni::load_ponni_tensor<2>(file,"test.input");
    auto expected = ponni::load_ponni_tensor<2>(file,"test.output");
    auto out = model.forward_batch_parallel(in);
    auto out_h = ponni::create_host_copy(out);
    auto exp_h = ponni::create_host_copy(expected);

    if (out_h.extent(0) != exp_h.extent(0) || out_h.extent(1) != exp_h.extent(1)) {
      std::cerr << "Projection skip integration output shape mismatch" << std::endl;
      ok = false;
    } else {
      for (int j = 0; j < out_h.extent(1); j++) {
        for (int i = 0; i < out_h.extent(0); i++) {
          float diff = std::abs(out_h(i,j) - exp_h(i,j));
          std::cout << "Absolute difference for Output(" << i << "," << j << "): " << diff << std::endl;
          if (diff > 1.e-5f) ok = false;
        }
      }
      if (!ok) std::cerr << "Projection skip integration output mismatch" << std::endl;
    }
  }

  Kokkos::finalize();

  if (!ok) return 1;
  std::cout << "projection_skip passed" << std::endl;
  return 0;
}
