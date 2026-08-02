#include "ponni.h"

#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {

bool nearly_equal(float a, float b, float tol = 1e-6f) {
  return std::fabs(a - b) <= tol;
}

}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ponni::init_device_pool(128ULL * 1024ULL * 1024ULL);

  bool ok = true;
  auto require_true = [&](bool cond, const std::string& msg) {
    if (!cond) {
      ok = false;
      std::cerr << "FAILED: " << msg << std::endl;
    }
  };

  {
    using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;
    using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;

    // Verify Initializer_None performs no write.
    real1d unchanged("unchanged", 8);
    Kokkos::deep_copy(unchanged, 3.0f);
    ponni::Initializer_None<float>().fill(unchanged);
    auto unchanged_host = ponni::create_host_copy(unchanged);
    for (int i = 0; i < unchanged_host.extent(0); ++i) {
      require_true(nearly_equal(unchanged_host(i), 3.0f), "Initializer_None should not modify values");
    }

    // Verify Random_Uniform stays within bounds.
    real1d rnd("rnd", 1024);
    ponni::Initializer_Random_Uniform<float> rand_init(-0.2f, 0.3f, 12345);
    rand_init.fill(rnd);
    auto rnd_host = ponni::create_host_copy(rnd);
    for (int i = 0; i < rnd_host.extent(0); ++i) {
      require_true(rnd_host(i) >= -0.2f && rnd_host(i) <= 0.3f,
                   "Initializer_Random_Uniform generated an out-of-range value");
    }

    // Build a small deterministic model and verify output.
    real2d weights("weights", 2, 2);
    real1d bias("bias", 2);
    Kokkos::deep_copy(weights, 1.0f);
    Kokkos::deep_copy(bias, 1.0f);

    auto model = ponni::create_inference_model(
        ponni::Matvec<float>(weights),
        ponni::Bias<float>(bias),
        ponni::Relu<float>(2, 0.1f));

    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> in_host("in_host", 2, 1);
    in_host(0, 0) = 1.0f;
    in_host(1, 0) = 2.0f;

    auto out_dev = model.forward_batch_parallel(ponni::create_device_copy(in_host));
    auto out_host = ponni::create_host_copy(out_dev);

    require_true(nearly_equal(out_host(0, 0), 4.0f), "Model output(0,0) should be 4.0");
    require_true(nearly_equal(out_host(1, 0), 4.0f), "Model output(1,0) should be 4.0");

    // Cover view reduction path.
    float out_sum = ponni::intrinsics::sum(out_dev);
    require_true(nearly_equal(out_sum, 8.0f), "intrinsics::sum(view) should equal 8.0");

    // Cover SArray reduction path.
    ponni::SArray<int, 4> ints;
    ints(0) = 1;
    ints(1) = 2;
    ints(2) = 3;
    ints(3) = 4;
    require_true(ponni::intrinsics::sum(ints) == 10, "intrinsics::sum(SArray) should equal 10");

    // Cover binary layer SArray compute paths.
    {
      using AddLayer = ponni::Binop_Add<0, float, 2>;
      ponni::SArray<float, 2> a;
      ponni::SArray<float, 2> b;
      ponni::SArray<float, 2> out;
      a(0) = 1.0f; a(1) = 2.0f;
      b(0) = 3.0f; b(1) = 4.0f;
      AddLayer::Params p{2, 2};
      AddLayer::compute_all_outputs(a, b, out, p);
      require_true(nearly_equal(out(0), 4.0f) && nearly_equal(out(1), 6.0f),
                   "Binop_Add SArray path produced incorrect output");
    }

    {
      using ConcatLayer = ponni::Binop_Concatenate<0, float, 2, 2>;
      ponni::SArray<float, 2> left;
      ponni::SArray<float, 2> right;
      ponni::SArray<float, 4> out;
      left(0) = 10.0f; left(1) = 20.0f;
      right(0) = 30.0f; right(1) = 40.0f;

      ConcatLayer::Params after_p{2, 4, true};
      ConcatLayer::compute_all_outputs(left, right, out, after_p);
      require_true(nearly_equal(out(0), 10.0f) && nearly_equal(out(1), 20.0f) &&
                   nearly_equal(out(2), 30.0f) && nearly_equal(out(3), 40.0f),
                   "Binop_Concatenate after=true path produced incorrect output");

      ConcatLayer::Params before_p{2, 4, false};
      ConcatLayer::compute_all_outputs(left, right, out, before_p);
      require_true(nearly_equal(out(0), 30.0f) && nearly_equal(out(1), 40.0f) &&
                   nearly_equal(out(2), 10.0f) && nearly_equal(out(3), 20.0f),
                   "Binop_Concatenate after=false path produced incorrect output");
    }

    // Cover Save_State SArray compute path.
    {
      using SaveLayer = ponni::Save_State<0, float, 3>;
      ponni::SArray<float, 3> in;
      ponni::SArray<float, 3> out;
      in(0) = 7.0f;
      in(1) = 8.0f;
      in(2) = 9.0f;
      SaveLayer::Params p{3, 3};
      SaveLayer::compute_all_outputs(in, out, p);
      require_true(nearly_equal(out(0), 7.0f) && nearly_equal(out(1), 8.0f) && nearly_equal(out(2), 9.0f),
                   "Save_State SArray path produced incorrect output");
    }
  }

  ponni::finalize_device_pool();
  Kokkos::finalize();

  if (!ok) return 1;
  std::cout << "core_unit passed" << std::endl;
  return 0;
}
