#include "ponni.h"

#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {

bool nearly_equal(float a, float b, float tol = 1e-6f) {
  return std::fabs(a - b) <= tol;
}

template <class Layer>
std::array<float,3> compute_activation_view(Layer const & layer) {
  using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;

  real2d input("activation_input", 3, 1);
  auto input_h = ponni::create_host_copy(input);
  input_h(0,0) = -1.0f;
  input_h(1,0) =  0.0f;
  input_h(2,0) =  2.0f;
  Kokkos::deep_copy(input, input_h);

  real2d output("activation_output", 3, 1);
  auto const params = layer.params;
  Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int ibatch) {
    Layer::compute_all_outputs(input, output, ibatch, params);
  });
  auto output_h = ponni::create_host_copy(output);
  return {output_h(0,0), output_h(1,0), output_h(2,0)};
}

template <int NIn, int NOut, class Layer>
Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>
compute_sarray_unary(Layer const & layer, std::array<float,NIn> const & values) {
  using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;

  real1d input("sarray_unary_input", NIn);
  auto input_h = ponni::create_host_copy(input);
  for (int i = 0; i < NIn; i++) input_h(i) = values[i];
  Kokkos::deep_copy(input, input_h);

  real1d output("sarray_unary_output", NOut);
  auto const params = layer.params;
  Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int) {
    ponni::SArray<float,NIn> input_s;
    ponni::SArray<float,NOut> output_s;
    for (int i = 0; i < NIn; i++) input_s(i) = input(i);
    Layer::compute_all_outputs(input_s, output_s, params);
    for (int i = 0; i < NOut; i++) output(i) = output_s(i);
  });
  return ponni::create_host_copy(output);
}

template <int N1, int N2, int NOut, class Layer>
Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>
compute_sarray_binary(Layer const & layer,
                      std::array<float,N1> const & values_1,
                      std::array<float,N2> const & values_2) {
  using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;

  real1d input_1("sarray_binary_input_1", N1);
  real1d input_2("sarray_binary_input_2", N2);
  auto input_1_h = ponni::create_host_copy(input_1);
  auto input_2_h = ponni::create_host_copy(input_2);
  for (int i = 0; i < N1; i++) input_1_h(i) = values_1[i];
  for (int i = 0; i < N2; i++) input_2_h(i) = values_2[i];
  Kokkos::deep_copy(input_1, input_1_h);
  Kokkos::deep_copy(input_2, input_2_h);

  real1d output("sarray_binary_output", NOut);
  auto const params = layer.params;
  Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int) {
    ponni::SArray<float,N1> input_1_s;
    ponni::SArray<float,N2> input_2_s;
    ponni::SArray<float,NOut> output_s;
    for (int i = 0; i < N1; i++) input_1_s(i) = input_1(i);
    for (int i = 0; i < N2; i++) input_2_s(i) = input_2(i);
    Layer::compute_all_outputs(input_1_s, input_2_s, output_s, params);
    for (int i = 0; i < NOut; i++) output(i) = output_s(i);
  });
  return ponni::create_host_copy(output);
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
        ponni::LeakyRelu<float>(2, 0.1f));

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

      ConcatLayer concat_layer(2, 4, false);
      require_true(std::string(concat_layer.get_label()) == "Binop_Concatenate",
                   "Binop_Concatenate label is incorrect");
      require_true(concat_layer.get_num_inputs() == 2 && concat_layer.get_num_outputs() == 4 &&
                   ConcatLayer::get_num_inputs(concat_layer.params) == 2 &&
                   ConcatLayer::get_num_outputs(concat_layer.params) == 4,
                   "Binop_Concatenate input/output size is incorrect");
      real1d no_concat_parameters;
      concat_layer.set_trainable_parameters(no_concat_parameters);
      require_true(concat_layer.get_num_trainable_parameters() == 0 &&
                   !concat_layer.get_trainable_parameters().is_allocated(),
                   "Binop_Concatenate trainable API is incorrect");
      auto concat_arr = concat_layer.to_array();
      ConcatLayer concat_reload;
      concat_reload.from_array(concat_arr);
      require_true(concat_reload.params.after == false,
           "Binop_Concatenate to_array/from_array should preserve after option");
      concat_reload.validate(2);
      require_true(concat_arr.extent(0) == concat_layer.get_array_representation_size(),
                   "Binop_Concatenate serialized size is incorrect");

      real2d left_view("concat_left", 2, 1);
      real2d right_view("concat_right", 2, 1);
      auto left_view_h = ponni::create_host_copy(left_view);
      auto right_view_h = ponni::create_host_copy(right_view);
      left_view_h(0,0) = left(0); left_view_h(1,0) = left(1);
      right_view_h(0,0) = right(0); right_view_h(1,0) = right(1);
      Kokkos::deep_copy(left_view, left_view_h);
      Kokkos::deep_copy(right_view, right_view_h);
      real2d concat_output("concat_output", 4, 1);
      auto const concat_params = concat_layer.params;
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int ibatch) {
        ConcatLayer::compute_all_outputs(left_view, right_view, concat_output, ibatch, concat_params);
      });
      auto concat_output_h = ponni::create_host_copy(concat_output);
      require_true(nearly_equal(concat_output_h(0,0), 30.0f) && nearly_equal(concat_output_h(3,0), 20.0f),
                   "Binop_Concatenate view output is incorrect");
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

      SaveLayer save_layer(3);
      require_true(std::string(save_layer.get_label()) == "Save_State", "Save_State label is incorrect");
      require_true(save_layer.get_num_inputs() == 3 && save_layer.get_num_outputs() == 3 &&
                   SaveLayer::get_num_inputs(save_layer.params) == 3 &&
                   SaveLayer::get_num_outputs(save_layer.params) == 3,
                   "Save_State input/output size is incorrect");
      real1d no_save_parameters;
      save_layer.set_trainable_parameters(no_save_parameters);
      require_true(save_layer.get_num_trainable_parameters() == 0 &&
                   !save_layer.get_trainable_parameters().is_allocated(),
                   "Save_State trainable API is incorrect");
      auto save_arr = save_layer.to_array();
      SaveLayer save_reload;
      save_reload.from_array(save_arr);
      save_reload.validate();
      require_true(save_reload.get_num_inputs() == 3, "Save_State to_array/from_array roundtrip failed");
      require_true(save_arr.extent(0) == save_layer.get_array_representation_size(),
                   "Save_State serialized size is incorrect");
    }

    // Cover layer-level API options: trainable flags, to_array/from_array, and set/get trainable parameters.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;

      real2d mv_w("mv_w", 2, 3);
      auto mv_w_h = ponni::create_host_copy(mv_w);
      mv_w_h(0,0) = 1.0f; mv_w_h(0,1) = 2.0f; mv_w_h(0,2) = 3.0f;
      mv_w_h(1,0) = 4.0f; mv_w_h(1,1) = 5.0f; mv_w_h(1,2) = 6.0f;
      Kokkos::deep_copy(mv_w, mv_w_h);

      ponni::Matvec<float> mv(mv_w, true);
      require_true(std::string(mv.get_label()) == "Matvec", "Matvec label is incorrect");
      require_true(mv.get_num_inputs() == 2 && mv.get_num_outputs() == 3,
                   "Matvec input/output size is incorrect");
      require_true(ponni::Matvec<float>::get_num_inputs(mv.params) == 2 &&
                   ponni::Matvec<float>::get_num_outputs(mv.params) == 3,
                   "Matvec static input/output size is incorrect");
      require_true(mv.get_num_trainable_parameters() == 6, "Matvec trainable count should be 6");

      real1d mv_new_params("mv_new_params", 6);
      auto mv_new_params_h = ponni::create_host_copy(mv_new_params);
      for (int i = 0; i < 6; ++i) mv_new_params_h(i) = 10.0f + static_cast<float>(i);
      Kokkos::deep_copy(mv_new_params, mv_new_params_h);
      mv.set_trainable_parameters(mv_new_params);
      auto mv_got = ponni::create_host_copy(mv.get_trainable_parameters());
      require_true(nearly_equal(mv_got(0), 10.0f) && nearly_equal(mv_got(5), 15.0f),
                   "Matvec set/get trainable parameters failed");

      auto mv_arr = mv.to_array();
      ponni::Matvec<float> mv_reload;
      mv_reload.from_array(mv_arr);
      mv_reload.validate();
      require_true(mv_reload.get_num_inputs() == 2 && mv_reload.get_num_outputs() == 3,
                   "Matvec to_array/from_array roundtrip failed");
      require_true(mv_arr.extent(0) == mv.get_array_representation_size(),
                   "Matvec serialized size is incorrect");

      using StaticMatvec = ponni::Matvec<float,2,3>;
      StaticMatvec static_mv(mv_w, true);
      auto mv_output_s = compute_sarray_unary<2,3>(static_mv, std::array<float,2>{1.0f, 2.0f});
      require_true(nearly_equal(mv_output_s(0), 36.0f) && nearly_equal(mv_output_s(2), 42.0f),
                   "Matvec SArray output is incorrect");

      ponni::Matvec<float> initialized_mv(2, 3, true, ponni::Initializer_Constant<float>(2.0f));
      initialized_mv.validate();
      auto initialized_mv_h = ponni::create_host_copy(initialized_mv.params.weights);
      require_true(nearly_equal(initialized_mv_h(0,0), 2.0f), "Matvec initializer constructor failed");

      ponni::Matvec<float> mv_not_trainable(mv_w, false);
      require_true(mv_not_trainable.get_num_trainable_parameters() == 0,
                   "Matvec non-trainable option should disable trainable parameters");
      require_true(!mv_not_trainable.get_trainable_parameters().is_allocated(),
                   "Matvec non-trainable get_trainable_parameters should be empty");

      real1d b_w("b_w", 3);
      auto b_w_h = ponni::create_host_copy(b_w);
      b_w_h(0) = 1.0f; b_w_h(1) = 2.0f; b_w_h(2) = 3.0f;
      Kokkos::deep_copy(b_w, b_w_h);

      ponni::Bias<float> bias(b_w, true);
      require_true(std::string(bias.get_label()) == "Bias", "Bias label is incorrect");
      require_true(bias.get_num_inputs() == 3 && bias.get_num_outputs() == 3,
                   "Bias input/output size is incorrect");
      require_true(ponni::Bias<float>::get_num_inputs(bias.params) == 3 &&
                   ponni::Bias<float>::get_num_outputs(bias.params) == 3,
                   "Bias static input/output size is incorrect");
      require_true(bias.get_num_trainable_parameters() == 3, "Bias trainable count should be 3");

      real1d b_new_params("b_new_params", 3);
      auto b_new_params_h = ponni::create_host_copy(b_new_params);
      b_new_params_h(0) = -1.0f; b_new_params_h(1) = -2.0f; b_new_params_h(2) = -3.0f;
      Kokkos::deep_copy(b_new_params, b_new_params_h);
      bias.set_trainable_parameters(b_new_params);
      auto b_got = ponni::create_host_copy(bias.get_trainable_parameters());
      require_true(nearly_equal(b_got(0), -1.0f) && nearly_equal(b_got(2), -3.0f),
                   "Bias set/get trainable parameters failed");

      auto bias_arr = bias.to_array();
      ponni::Bias<float> bias_reload;
      bias_reload.from_array(bias_arr);
      bias_reload.validate();
      require_true(bias_reload.get_num_inputs() == 3, "Bias to_array/from_array roundtrip failed");
      require_true(bias_arr.extent(0) == bias.get_array_representation_size(),
                   "Bias serialized size is incorrect");

      using StaticBias = ponni::Bias<float,3>;
      StaticBias static_bias(b_w, true);
      auto bias_output_s = compute_sarray_unary<3,3>(static_bias, std::array<float,3>{1.0f, 2.0f, 3.0f});
      require_true(nearly_equal(bias_output_s(0), 0.0f) && nearly_equal(bias_output_s(2), 0.0f),
                   "Bias SArray output is incorrect");

      ponni::Bias<float> initialized_bias(3, true, ponni::Initializer_Constant<float>(4.0f));
      initialized_bias.validate();
      auto initialized_bias_h = ponni::create_host_copy(initialized_bias.params.weights);
      require_true(nearly_equal(initialized_bias_h(2), 4.0f), "Bias initializer constructor failed");

      ponni::Bias<float> bias_not_trainable(b_w, false);
      require_true(bias_not_trainable.get_num_trainable_parameters() == 0,
                   "Bias non-trainable option should disable trainable parameters");
      require_true(!bias_not_trainable.get_trainable_parameters().is_allocated(),
                   "Bias non-trainable get_trainable_parameters should be empty");

      using AddLayer = ponni::Binop_Add<0, float, 3>;
      AddLayer add_layer(3);
      require_true(std::string(add_layer.get_label()) == "Binop_Add", "Binop_Add label is incorrect");
      require_true(AddLayer::get_num_inputs(add_layer.params) == 3 &&
                   AddLayer::get_num_outputs(add_layer.params) == 3,
                   "Binop_Add static input/output size is incorrect");
      real1d no_add_parameters;
      add_layer.set_trainable_parameters(no_add_parameters);
      require_true(add_layer.get_num_trainable_parameters() == 0 &&
                   !add_layer.get_trainable_parameters().is_allocated(),
                   "Binop_Add trainable API is incorrect");
      auto add_arr = add_layer.to_array();
      AddLayer add_reload;
      add_reload.from_array(add_arr);
      add_reload.validate(3);
      require_true(add_reload.get_num_outputs() == 3, "Binop_Add to_array/from_array roundtrip failed");
      require_true(add_arr.extent(0) == add_layer.get_array_representation_size(),
                   "Binop_Add serialized size is incorrect");

      using Proj = ponni::Binop_Projection_Add<0, float, 3, 2>;
      real2d proj_w_nt("proj_w_nt", 2, 3);
      real1d proj_b_nt("proj_b_nt", 3);
      Kokkos::deep_copy(proj_w_nt, 0.0f);
      Kokkos::deep_copy(proj_b_nt, 0.0f);
      Proj proj_nt(proj_w_nt, proj_b_nt, false);
      require_true(proj_nt.get_num_trainable_parameters() == 0,
                   "Projection skip non-trainable option should disable trainable parameters");
      require_true(!proj_nt.get_trainable_parameters().is_allocated(),
                   "Projection skip non-trainable get_trainable_parameters should be empty");
      auto proj_nt_arr = proj_nt.to_array();
      Proj proj_nt_reload;
      proj_nt_reload.from_array(proj_nt_arr);
      proj_nt_reload.validate(2);
      require_true(proj_nt_reload.get_num_trainable_parameters() == 0,
                   "Projection skip to_array/from_array should preserve non-trainable option");
    }

    // Cover every activation's host API plus its SArray and DeviceSpace compute paths.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;

      auto check_activation = [&]<class Layer>(Layer layer,
                                                std::array<float,3> const & expected,
                                                std::string const & name,
                                                float tolerance = 1.e-5f) {
        layer.validate();
        require_true(std::string(layer.get_label()) == name, name + " label is incorrect");
        require_true(layer.get_num_inputs() == 3 && layer.get_num_outputs() == 3,
                     name + " input/output size is incorrect");
        require_true(Layer::get_num_inputs(layer.params) == 3 && Layer::get_num_outputs(layer.params) == 3,
                     name + " static input/output size is incorrect");
        require_true(layer.get_num_trainable_parameters() == 0, name + " should not have trainable parameters");

        real1d no_parameters;
        layer.set_trainable_parameters(no_parameters);
        require_true(!layer.get_trainable_parameters().is_allocated(),
                     name + " get_trainable_parameters should be empty");

        auto data = layer.to_array();
        require_true(data.extent(0) == layer.get_array_representation_size(),
                     name + " serialized size is incorrect");
        Layer reloaded;
        reloaded.from_array(data);
        reloaded.validate();
        auto reloaded_data = reloaded.to_array();
        require_true(reloaded_data.extent(0) == data.extent(0), name + " roundtrip size changed");
        for (int i = 0; i < data.extent(0); i++) {
          require_true(nearly_equal(reloaded_data(i), data(i)), name + " roundtrip data changed");
        }

        ponni::SArray<float,3> input_s;
        ponni::SArray<float,3> output_s;
        input_s(0) = -1.0f;
        input_s(1) =  0.0f;
        input_s(2) =  2.0f;
        Layer::compute_all_outputs(input_s, output_s, layer.params);
        for (int i = 0; i < 3; i++) {
          require_true(nearly_equal(output_s(i), expected[i], tolerance), name + " SArray output is incorrect");
        }

        auto const output_h = compute_activation_view(layer);
        for (int i = 0; i < 3; i++) {
          require_true(nearly_equal(output_h[i], expected[i], tolerance), name + " view output is incorrect");
        }
      };

      float constexpr selu_alpha = 1.6732632423543772848f;
      float constexpr selu_scale = 1.0507009873554804934f;
      float const softmax_sum = std::exp(-3.0f) + std::exp(-2.0f) + 1.0f;
      float const log_sum_exp = std::log(softmax_sum) + 2.0f;

      check_activation(ponni::Relu<float,3>(3), {0.0f, 0.0f, 2.0f}, "ReLU");
      check_activation(ponni::LeakyRelu<float,3>(3, 0.1f), {-0.1f, 0.0f, 2.0f}, "LeakyReLU");
      check_activation(ponni::Elu<float,3>(3), {std::exp(-1.0f) - 1.0f, 0.0f, 2.0f}, "ELU");
      check_activation(ponni::Selu<float,3>(3),
                       {selu_scale * selu_alpha * (std::exp(-1.0f) - 1.0f), 0.0f, selu_scale * 2.0f},
                       "SELU", 2.e-5f);
      check_activation(ponni::Gelu<float,3>(3),
                       {-0.5f * (1.0f - std::erf(1.0f / std::sqrt(2.0f))), 0.0f,
                         1.0f * (1.0f + std::erf(std::sqrt(2.0f)))},
                       "GELU", 2.e-5f);
      check_activation(ponni::Gelu<float,3>(3, true),
                       {-0.158808f, 0.0f, 1.954598f}, "GELU", 2.e-5f);
      check_activation(ponni::Silu<float,3>(3),
                       {-1.0f / (1.0f + std::exp(1.0f)), 0.0f, 2.0f / (1.0f + std::exp(-2.0f))},
                       "SiLU", 2.e-5f);
      check_activation(ponni::Sigmoid<float,3>(3),
                       {1.0f / (1.0f + std::exp(1.0f)), 0.5f, 1.0f / (1.0f + std::exp(-2.0f))},
                       "Sigmoid", 2.e-5f);
      check_activation(ponni::Tanh<float,3>(3), {std::tanh(-1.0f), 0.0f, std::tanh(2.0f)}, "Tanh");
      check_activation(ponni::Softmax<float,3>(3),
                       {std::exp(-3.0f) / softmax_sum, std::exp(-2.0f) / softmax_sum, 1.0f / softmax_sum},
                       "Softmax");
      check_activation(ponni::LogSoftmax<float,3>(3),
                       {-1.0f - log_sum_exp, -log_sum_exp, 2.0f - log_sum_exp}, "LogSoftmax");
      check_activation(ponni::Softplus<float,3>(3, 2.0f),
                       {0.5f * std::log(1.0f + std::exp(-2.0f)), 0.5f * std::log(2.0f),
                        0.5f * std::log(1.0f + std::exp(4.0f))},
                       "Softplus", 2.e-5f);
      check_activation(ponni::HardSigmoid<float,3>(3, 0.2f, 0.5f), {0.3f, 0.5f, 0.9f}, "HardSigmoid");
      check_activation(ponni::HardSwish<float,3>(3), {-2.0f / 6.0f, 0.0f, 10.0f / 6.0f}, "HardSwish");
      check_activation(ponni::Mish<float,3>(3),
                       {-std::tanh(std::log(1.0f + std::exp(-1.0f))), 0.0f,
                         2.0f * std::tanh(std::log(1.0f + std::exp(2.0f)))},
                       "Mish", 2.e-5f);
    }

    // Cover LayerNorm forward path plus trainable parameter set/get and serialization.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;

      real1d gamma("ln_gamma", 4);
      real1d beta("ln_beta", 4);
      Kokkos::deep_copy(gamma, 1.0f);
      Kokkos::deep_copy(beta, 0.0f);
      ponni::LayerNorm<float> ln(gamma, beta, 1.e-5f, true);
      require_true(std::string(ln.get_label()) == "LayerNorm", "LayerNorm label is incorrect");
      require_true(ln.get_num_inputs() == 4 && ln.get_num_outputs() == 4 &&
                   ponni::LayerNorm<float>::get_num_inputs(ln.params) == 4 &&
                   ponni::LayerNorm<float>::get_num_outputs(ln.params) == 4,
                   "LayerNorm input/output size is incorrect");
      require_true(ln.get_num_trainable_parameters() == 8, "LayerNorm trainable count is incorrect");

      real2d ln_in("ln_in", 4, 1);
      auto ln_in_h = ponni::create_host_copy(ln_in);
      ln_in_h(0,0) = 1.0f;
      ln_in_h(1,0) = 2.0f;
      ln_in_h(2,0) = 3.0f;
      ln_in_h(3,0) = 4.0f;
      Kokkos::deep_copy(ln_in, ln_in_h);

      real2d ln_out("ln_out", 4, 1);
      auto const ln_kernel_params = ln.params;
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int ibatch) {
        ponni::LayerNorm<float>::compute_all_outputs(ln_in, ln_out, ibatch, ln_kernel_params);
      });
      auto ln_out_h = ponni::create_host_copy(ln_out);
      float mean = 0.25f * (ln_out_h(0,0) + ln_out_h(1,0) + ln_out_h(2,0) + ln_out_h(3,0));
      require_true(nearly_equal(mean, 0.0f, 1e-4f), "LayerNorm output mean should be near zero");

      using StaticLayerNorm = ponni::LayerNorm<float,4>;
      StaticLayerNorm static_ln(gamma, beta, 1.e-5f, true);
      auto ln_output_s = compute_sarray_unary<4,4>(static_ln, std::array<float,4>{1.0f, 2.0f, 3.0f, 4.0f});
      float const ln_s_mean = 0.25f * (ln_output_s(0) + ln_output_s(1) + ln_output_s(2) + ln_output_s(3));
      require_true(nearly_equal(ln_s_mean, 0.0f, 1.e-4f), "LayerNorm SArray output mean should be near zero");

      real1d ln_params("ln_params", 8);
      auto ln_params_h = ponni::create_host_copy(ln_params);
      for (int i = 0; i < 4; ++i) ln_params_h(i) = 2.0f;
      for (int i = 0; i < 4; ++i) ln_params_h(4 + i) = 0.5f;
      Kokkos::deep_copy(ln_params, ln_params_h);
      ln.set_trainable_parameters(ln_params);
      auto got_ln_params = ln.get_trainable_parameters();
      auto got_ln_params_h = ponni::create_host_copy(got_ln_params);
      require_true(nearly_equal(got_ln_params_h(0), 2.0f), "LayerNorm set/get trainable parameters failed");
      require_true(nearly_equal(got_ln_params_h(4), 0.5f), "LayerNorm set/get trainable parameters failed for beta");

      auto ln_arr = ln.to_array();
      ponni::LayerNorm<float> ln_reload;
      ln_reload.from_array(ln_arr);
      ln_reload.validate();
      require_true(ln_reload.get_num_inputs() == 4, "LayerNorm to_array/from_array roundtrip failed");
      require_true(ln_arr.extent(0) == ln.get_array_representation_size(),
                   "LayerNorm serialized size is incorrect");

      ponni::LayerNorm<float> initialized_ln(4);
      initialized_ln.validate();
      auto initialized_ln_gamma_h = ponni::create_host_copy(initialized_ln.params.gamma);
      auto initialized_ln_beta_h = ponni::create_host_copy(initialized_ln.params.beta);
      require_true(nearly_equal(initialized_ln_gamma_h(0), 1.0f) && nearly_equal(initialized_ln_beta_h(3), 0.0f),
                   "LayerNorm size constructor failed");

      ponni::LayerNorm<float> ln_not_trainable(gamma, beta, 1.e-5f, false);
      ln_not_trainable.set_trainable_parameters(ln_params);
      require_true(ln_not_trainable.get_num_trainable_parameters() == 0,
                   "LayerNorm non-trainable option should disable trainable parameters");
      require_true(!ln_not_trainable.get_trainable_parameters().is_allocated(),
                   "LayerNorm non-trainable get_trainable_parameters should be empty");
    }

    // Cover MinMaxNorm forward path and serialization.
    {
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;
      ponni::MinMaxNorm<float> mm(3, -1.0f, 1.0f);
      require_true(std::string(mm.get_label()) == "MinMaxNorm", "MinMaxNorm label is incorrect");
      require_true(mm.get_num_inputs() == 3 && mm.get_num_outputs() == 3 &&
                   ponni::MinMaxNorm<float>::get_num_inputs(mm.params) == 3 &&
                   ponni::MinMaxNorm<float>::get_num_outputs(mm.params) == 3,
                   "MinMaxNorm input/output size is incorrect");
      Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace> no_mm_parameters;
      mm.set_trainable_parameters(no_mm_parameters);
      require_true(mm.get_num_trainable_parameters() == 0 &&
                   !mm.get_trainable_parameters().is_allocated(),
                   "MinMaxNorm trainable API is incorrect");
      real2d mm_in("mm_in", 3, 1);
      auto mm_in_h = ponni::create_host_copy(mm_in);
      mm_in_h(0,0) = 2.0f;
      mm_in_h(1,0) = 5.0f;
      mm_in_h(2,0) = 8.0f;
      Kokkos::deep_copy(mm_in, mm_in_h);
      real2d mm_out("mm_out", 3, 1);
      auto const mm_params = mm.params;
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int ibatch) {
        ponni::MinMaxNorm<float>::compute_all_outputs(mm_in, mm_out, ibatch, mm_params);
      });
      auto mm_out_h = ponni::create_host_copy(mm_out);
      require_true(nearly_equal(mm_out_h(0,0), -1.0f, 1e-5f), "MinMaxNorm min value incorrect");
      require_true(nearly_equal(mm_out_h(2,0),  1.0f, 1e-5f), "MinMaxNorm max value incorrect");

      using StaticMinMaxNorm = ponni::MinMaxNorm<float,3>;
      StaticMinMaxNorm static_mm(3, -1.0f, 1.0f);
      ponni::SArray<float,3> mm_input_s;
      ponni::SArray<float,3> mm_output_s;
      mm_input_s(0) = 2.0f; mm_input_s(1) = 5.0f; mm_input_s(2) = 8.0f;
      StaticMinMaxNorm::compute_all_outputs(mm_input_s, mm_output_s, static_mm.params);
      require_true(nearly_equal(mm_output_s(0), -1.0f) && nearly_equal(mm_output_s(2), 1.0f),
                   "MinMaxNorm SArray output is incorrect");

      auto mm_arr = mm.to_array();
      ponni::MinMaxNorm<float> mm_reload;
      mm_reload.from_array(mm_arr);
      mm_reload.validate();
      require_true(mm_reload.get_num_inputs() == 3, "MinMaxNorm to_array/from_array roundtrip failed");
      require_true(mm_arr.extent(0) == mm.get_array_representation_size(),
                   "MinMaxNorm serialized size is incorrect");
    }

    // Cover projection skip layer including trainable parameters and serialization.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, ponni::DeviceSpace>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;
      real2d proj_w("proj_w", 2, 3);
      real1d proj_b("proj_b", 3);
      auto proj_w_h = ponni::create_host_copy(proj_w);
      auto proj_b_h = ponni::create_host_copy(proj_b);
      proj_w_h(0,0) = 1.0f; proj_w_h(0,1) = 0.0f; proj_w_h(0,2) = 0.5f;
      proj_w_h(1,0) = 0.0f; proj_w_h(1,1) = 1.0f; proj_w_h(1,2) = 0.5f;
      proj_b_h(0) = 0.1f; proj_b_h(1) = 0.2f; proj_b_h(2) = 0.3f;
      Kokkos::deep_copy(proj_w, proj_w_h);
      Kokkos::deep_copy(proj_b, proj_b_h);

      using Proj = ponni::Binop_Projection_Add<0, float, 3, 2>;
      Proj proj(proj_w, proj_b, true);
      require_true(std::string(proj.get_label()) == "Binop_Projection_Add",
                   "Projection skip label is incorrect");
      require_true(proj.get_num_inputs() == 3 && proj.get_num_outputs() == 3 &&
                   Proj::get_num_inputs(proj.params) == 3 && Proj::get_num_outputs(proj.params) == 3,
                   "Projection skip input/output size is incorrect");

      real2d cur("cur", 3, 1);
      real2d sav("sav", 2, 1);
      auto cur_h = ponni::create_host_copy(cur);
      auto sav_h = ponni::create_host_copy(sav);
      cur_h(0,0) = 1.0f; cur_h(1,0) = 2.0f; cur_h(2,0) = 3.0f;
      sav_h(0,0) = 4.0f; sav_h(1,0) = 5.0f;
      Kokkos::deep_copy(cur, cur_h);
      Kokkos::deep_copy(sav, sav_h);
      real2d out("proj_out", 3, 1);
      auto const proj_params = proj.params;
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int ibatch) {
        Proj::compute_all_outputs(cur, sav, out, ibatch, proj_params);
      });
      auto out_h = ponni::create_host_copy(out);
      require_true(nearly_equal(out_h(0,0), 5.1f, 1e-5f), "Projection skip output(0) incorrect");
      require_true(nearly_equal(out_h(1,0), 7.2f, 1e-5f), "Projection skip output(1) incorrect");
      require_true(nearly_equal(out_h(2,0), 7.8f, 1e-5f), "Projection skip output(2) incorrect");

      auto proj_output_s = compute_sarray_binary<3,2,3>(
          proj, std::array<float,3>{1.0f, 2.0f, 3.0f}, std::array<float,2>{4.0f, 5.0f});
      require_true(nearly_equal(proj_output_s(0), 5.1f) && nearly_equal(proj_output_s(2), 7.8f),
                   "Projection skip SArray output is incorrect");

      auto proj_trainable = proj.get_trainable_parameters();
      require_true(proj_trainable.extent(0) == 9, "Projection skip trainable parameter count incorrect");
      proj.set_trainable_parameters(proj_trainable);

      auto proj_arr = proj.to_array();
      Proj proj_reload;
      proj_reload.from_array(proj_arr);
      proj_reload.validate(2);
      require_true(proj_reload.get_num_outputs() == 3, "Projection skip to_array/from_array roundtrip failed");
      require_true(proj_arr.extent(0) == proj.get_array_representation_size(),
                   "Projection skip serialized size is incorrect");

      Proj initialized_proj(3, 2, true, ponni::Initializer_Constant<float>(0.25f));
      initialized_proj.validate(2);
      auto initialized_proj_weights_h = ponni::create_host_copy(initialized_proj.params.weights);
      auto initialized_proj_bias_h = ponni::create_host_copy(initialized_proj.params.bias);
      require_true(nearly_equal(initialized_proj_weights_h(1,2), 0.25f) &&
                   nearly_equal(initialized_proj_bias_h(2), 0.25f),
                   "Projection skip initializer constructor failed");
    }

    // Cover advanced initializer suite with simple sanity checks.
    {
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, ponni::DeviceSpace>;
      real2d x("init_x", 8, 6);

      ponni::Initializer_Zeros<float>().fill(x);
      auto x_h = ponni::create_host_copy(x);
      require_true(nearly_equal(x_h(0,0), 0.0f), "Initializer_Zeros failed");

      ponni::Initializer_Ones<float>().fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(nearly_equal(x_h(0,0), 1.0f), "Initializer_Ones failed");

      ponni::Initializer_Constant<float>(2.5f).fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(nearly_equal(x_h(3,4), 2.5f), "Initializer_Constant failed");

      ponni::Initializer_Random_Normal<float>(0.0f, 0.2f, 111).fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(std::isfinite(x_h(2,2)), "Initializer_Random_Normal produced non-finite value");

      ponni::Initializer_Truncated_Normal<float>(0.0f, 0.1f, 222).fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(std::abs(x_h(1,1)) < 1.0f, "Initializer_Truncated_Normal produced outlier value");

      ponni::Initializer_Xavier_Uniform<float>(333).fill(x);
      ponni::Initializer_Xavier_Normal<float>(444).fill(x);
      ponni::Initializer_He_Uniform<float>(555).fill(x);
      ponni::Initializer_He_Normal<float>(666).fill(x);
      ponni::Initializer_Lecun_Uniform<float>(777).fill(x);
      ponni::Initializer_Lecun_Normal<float>(888).fill(x);
      ponni::Initializer_Random_Uniform<float>(-0.5f, 0.5f, 4321).fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(std::isfinite(x_h(0,1)) && std::isfinite(x_h(7,5)), "Variance-scaled initializer produced non-finite value");

      ponni::Initializer_Orthogonal<float>(1.0f, 999).fill(x);
      x_h = ponni::create_host_copy(x);
      require_true(std::isfinite(x_h(0,0)), "Initializer_Orthogonal produced non-finite value");
    }

    // Exercise both lanes and the packed multiply-add on the active device backend.
    {
      Kokkos::View<float*,ponni::DeviceSpace> result("two_half_result", 16);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 1, KOKKOS_LAMBDA(int) {
        ponni::TwoHalf const left = ponni::TwoHalf::from_floats(2.0f, -3.0f);
        ponni::TwoHalf const right = ponni::TwoHalf::from_floats(4.0f, 5.0f);
        ponni::TwoHalf const added = left + right;
        ponni::TwoHalf const fused = ponni::TwoHalf::fma(left, right, ponni::TwoHalf::from_floats(1.0f, 2.0f));
        ponni::TwoHalf const rounded_even = ponni::TwoHalf::round(ponni::TwoHalf::from_floats(2.5f, -3.5f));
        ponni::TwoHalf const rounded_odd = ponni::TwoHalf::round(ponni::TwoHalf::from_floats(1.5f, -2.5f));
        ponni::TwoHalf const signed_values = ponni::TwoHalf::sign(
            ponni::TwoHalf::from_floats(Kokkos::Experimental::quiet_NaN_v<float>, -0.0f));
        ponni::TwoMask const greater = ponni::TwoHalf::greater(left, right);
        ponni::TwoMask const either = ponni::TwoMask::logical_or(
            greater, ponni::TwoMask::from_bools(true, false));
        ponni::TwoMask const inverted = ponni::TwoMask::logical_not(either);
        ponni::TwoHalf const selected = ponni::TwoHalf::select(inverted, left, right);
        result(0) = added.low();
        result(1) = added.high();
        result(2) = fused.low();
        result(3) = fused.high();
        result(4) = rounded_even.low();
        result(5) = rounded_even.high();
        result(6) = rounded_odd.low();
        result(7) = rounded_odd.high();
        result(8) = signed_values.low();
        result(9) = signed_values.high();
        result(10) = greater.low() ? 1.0f : 0.0f;
        result(11) = greater.high() ? 1.0f : 0.0f;
        result(12) = inverted.low() ? 1.0f : 0.0f;
        result(13) = inverted.high() ? 1.0f : 0.0f;
        result(14) = selected.low();
        result(15) = selected.high();
      });
      auto result_h = ponni::create_host_copy(result);
      require_true(nearly_equal(result_h(0), 6.0f) && nearly_equal(result_h(1), 2.0f) &&
                   nearly_equal(result_h(2), 9.0f) && nearly_equal(result_h(3), -13.0f),
                   "TwoHalf packed arithmetic produced incorrect lanes");
      require_true(nearly_equal(result_h(4), 2.0f) && nearly_equal(result_h(5), -4.0f) &&
                   nearly_equal(result_h(6), 2.0f) && nearly_equal(result_h(7), -2.0f),
                   "TwoHalf ONNX ties-to-even rounding produced incorrect lanes");
      require_true(std::isnan(result_h(8)) && result_h(9) == 0.0f && !std::signbit(result_h(9)),
                   "TwoHalf ONNX sign handling produced incorrect lanes");
      require_true(nearly_equal(result_h(10), 0.0f) && nearly_equal(result_h(11), 0.0f) &&
                   nearly_equal(result_h(12), 0.0f) && nearly_equal(result_h(13), 1.0f) &&
                   nearly_equal(result_h(14), 4.0f) && nearly_equal(result_h(15), -3.0f),
                   "TwoMask comparison, logical, and selection operations produced incorrect lanes");
    }
  }

  ponni::finalize_device_pool();
  Kokkos::finalize();

  if (!ok) return 1;
  std::cout << "core_unit passed" << std::endl;
  return 0;
}
