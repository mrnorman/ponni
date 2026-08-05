#include "ponni.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

bool nearly_equal(float a, float b, float tol = 1e-6f) {
  return std::fabs(a - b) <= tol;
}

bool is_ieee_nan(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "NaN test requires a 32-bit float");
  static_assert(std::numeric_limits<float>::is_iec559,
                "NaN test requires IEEE-754 floating point");
  std::uint32_t const bits = std::bit_cast<std::uint32_t>(value);
  return (bits & 0x7f800000u) == 0x7f800000u &&
         (bits & 0x007fffffu) != 0;
}

struct SampleStats {
  double mean;
  double variance;
  double minimum;
  double maximum;
};

template <class ViewType>
SampleStats sample_stats(ViewType const & values) {
  auto const host = ponni::create_host_copy(values);
  double sum = 0;
  double square_sum = 0;
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < host.size(); i++) {
    double const value = static_cast<double>(host.data()[i]);
    sum += value;
    square_sum += value * value;
    minimum = value < minimum ? value : minimum;
    maximum = value > maximum ? value : maximum;
  }
  double const mean = sum / static_cast<double>(host.size());
  return {mean, square_sum / static_cast<double>(host.size()) - mean * mean, minimum, maximum};
}

template <class ViewType>
bool exactly_equal(ViewType const & left, ViewType const & right) {
  auto const left_host = ponni::create_host_copy(left);
  auto const right_host = ponni::create_host_copy(right);
  if (left_host.size() != right_host.size()) return false;
  for (std::size_t i = 0; i < left_host.size(); i++) {
    if (left_host.data()[i] != right_host.data()[i]) return false;
  }
  return true;
}

template <class Function>
bool throws_invalid_argument(Function const & function) {
  try {
    function();
  } catch (std::invalid_argument const &) {
    return true;
  }
  return false;
}

template <class Layer>
std::array<float,3> compute_activation_view(Layer const & layer) {
  using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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
  using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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
  using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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

  bool ok = true;
  auto require_true = [&](bool cond, const std::string& msg) {
    if (!cond) {
      ok = false;
      std::cerr << "FAILED: " << msg << std::endl;
    }
  };

  {
    using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
    using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

    // The host-only JSON parser is intentionally tiny, but still implements
    // nested values, Unicode escapes, padded headers, and duplicate rejection.
    {
      std::string const json = R"json({"name":"PONNI \u03c0","shape":[2,3],"valid":true}   )json";
      ponni::detail::JsonValue value;
      ponni::detail::JsonParser parser(json.data(),json.size());
      std::string error;
      require_true(parser.parse(value,error) && value.find("shape") != nullptr,
                   "PONNI JSON parser rejected a valid padded header: " + error);

      std::string const duplicate = R"({"tensor":1,"tensor":2})";
      ponni::detail::JsonParser duplicate_parser(duplicate.data(),duplicate.size());
      require_true(!duplicate_parser.parse(value,error) && error.find("duplicate") != std::string::npos,
                   "PONNI JSON parser should reject duplicate object keys");
    }

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
    require_true(!model.params.tmp1.is_allocated() && !model.params.tmp2.is_allocated(),
                 "A single fused dense block should write directly to output without scratch Views");

    // The tuple-derived fingerprint binds template weights to this precise
    // sequence of layer dimensions. Saving and loading use the same augmented
    // Safetensors contract as generated models, without involving ONNX.
    {
      std::string const path = "core_template_weights.ponni";
      std::string error;
      require_true(model.save_weights(path,&error), "Template weight save failed: " + error);
      real1d zero_parameters("zero_parameters", model.get_num_trainable_parameters());
      Kokkos::deep_copy(zero_parameters,0.0f);
      model.set_trainable_parameters(zero_parameters);
      require_true(model.load_weights(path,&error), "Template weight load failed: " + error);
      auto restored_output = ponni::create_host_copy(
          model.forward_batch_parallel(ponni::create_device_copy(in_host)));
      require_true(nearly_equal(restored_output(0,0),4.0f) && nearly_equal(restored_output(1,0),4.0f),
                   "Template PONNI-file round trip did not restore learned parameters");
      std::remove(path.c_str());
    }

    // The default model starts without batch scratch, grows on demand, keeps
    // larger capacity for reuse, and permits an explicit exact shrink.
    require_true(model.internal_state_capacity() == 1,
                 "First batch inference should allocate one scratch column");
    real2d larger_input("larger_input", 2, 4);
    Kokkos::deep_copy(larger_input, 1.0f);
    auto larger_output = model.forward_batch_parallel(larger_input);
    require_true(model.internal_state_capacity() == 4,
                 "Model scratch should grow to fit a larger batch");
    model.forward_batch_parallel(ponni::create_device_copy(in_host));
    require_true(model.internal_state_capacity() == 4,
                 "A smaller batch should retain existing scratch capacity");
    model.reallocate_internal_state(2);
    require_true(model.internal_state_capacity() == 2,
                 "Explicit internal-state reallocation should shrink capacity");
    require_true(larger_output.extent(1) == 4, "Larger batch output should retain its requested extent");

    // Exercise a dynamically shaped four-block network. The block widths are
    // deliberately asymmetric: intermediate widths 5, 3, and 7 assign tmp1 a
    // capacity of 7 and tmp2 a capacity of 3. Bias, parameter-free activations,
    // and parameterized activations are all folded into the dense output loops.
    {
      real2d w1("fused_w1", 2, 5);
      real2d w2("fused_w2", 5, 3);
      real2d w3("fused_w3", 3, 7);
      real2d w4("fused_w4", 7, 2);
      real1d b1("fused_b1", 5);
      real1d b2("fused_b2", 3);
      real1d b3("fused_b3", 7);
      real1d b4("fused_b4", 2);
      Kokkos::deep_copy(w1, 1.0f);
      Kokkos::deep_copy(w2, 1.0f);
      Kokkos::deep_copy(w3, 1.0f);
      Kokkos::deep_copy(w4, 1.0f);
      Kokkos::deep_copy(b1,  0.5f);
      Kokkos::deep_copy(b2, -0.25f);
      Kokkos::deep_copy(b3,  0.1f);
      Kokkos::deep_copy(b4, -0.2f);

      auto fused_model = ponni::create_inference_model(
          ponni::Matvec<float>(w1), ponni::Bias<float>(b1), ponni::Relu<float>(5), ponni::Tanh<float>(5),
          ponni::Matvec<float>(w2), ponni::Bias<float>(b2), ponni::Sigmoid<float>(3),
          ponni::Matvec<float>(w3), ponni::Bias<float>(b3), ponni::LeakyRelu<float>(7, 0.2f),
          ponni::Matvec<float>(w4), ponni::Bias<float>(b4), ponni::HardSwish<float>(2));

      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> fused_input_h("fused_input_h", 2, 3);
      fused_input_h(0,0) =  1.0f; fused_input_h(1,0) =  2.0f;
      fused_input_h(0,1) = -2.0f; fused_input_h(1,1) =  0.5f;
      fused_input_h(0,2) =  0.0f; fused_input_h(1,2) =  0.0f;
      auto fused_input = ponni::create_device_copy(fused_input_h);
      auto fused_output = fused_model.forward_batch_parallel(fused_input);
      auto fused_output_h = ponni::create_host_copy(fused_output);

      // The same block traversal is available to callers already inside a
      // larger Kokkos kernel, using the workspace allocated by the batch call.
      real2d fused_output_in_kernel("fused_output_in_kernel", 2, 3);
      auto const fused_params = fused_model.params;
      using FusedModel = decltype(fused_model);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 3, KOKKOS_LAMBDA(int ibatch) {
        FusedModel::forward_batch_parallel_in_kernel(fused_input, fused_output_in_kernel, fused_params, ibatch);
      });
      auto fused_output_in_kernel_h = ponni::create_host_copy(fused_output_in_kernel);

      require_true(fused_model.params.tmp1.extent(0) == 7 && fused_model.params.tmp1.extent(1) == 3,
                   "Fused tmp1 should use the largest even intermediate block width");
      require_true(fused_model.params.tmp2.extent(0) == 3 && fused_model.params.tmp2.extent(1) == 3,
                   "Fused tmp2 should use the largest odd intermediate block width");

      for (int ibatch = 0; ibatch < 3; ibatch++) {
        float const input_sum = fused_input_h(0,ibatch) + fused_input_h(1,ibatch);
        float const hidden1 = std::tanh(std::max(0.0f, input_sum + 0.5f));
        float const dense2 = 5.0f * hidden1 - 0.25f;
        float const hidden2 = 1.0f / (1.0f + std::exp(-dense2));
        float const dense3 = 3.0f * hidden2 + 0.1f;
        float const hidden3 = dense3 > 0.0f ? dense3 : 0.2f * dense3;
        float const dense4 = 7.0f * hidden3 - 0.2f;
        float const expected = dense4 <= -3.0f ? 0.0f :
                               dense4 >=  3.0f ? dense4 : dense4 * (dense4 + 3.0f) / 6.0f;
        for (int feature = 0; feature < 2; feature++) {
          require_true(nearly_equal(fused_output_h(feature,ibatch), expected, 1e-4f),
                       "Fused dynamically shaped network produced an incorrect output");
          require_true(nearly_equal(fused_output_in_kernel_h(feature,ibatch), expected, 1e-4f),
                       "Intra-kernel fused dynamically shaped network produced an incorrect output");
        }
      }
    }

    // A cross-feature barrier must end one fused region without disabling
    // fusion later in the tuple. This model plans two dense/pointwise blocks
    // around an in-place Softmax and needs only tmp1 for the materialized state.
    {
      real2d w1("mixed_w1", 2, 3);
      real2d w2("mixed_w2", 3, 2);
      real1d b1("mixed_b1", 3);
      real1d b2("mixed_b2", 2);
      Kokkos::deep_copy(w1, 1.0f);
      Kokkos::deep_copy(w2, 1.0f);
      auto b1_h = Kokkos::create_mirror_view(b1);
      auto b2_h = Kokkos::create_mirror_view(b2);
      b1_h(0) = 0.0f; b1_h(1) = 1.0f; b1_h(2) = 2.0f;
      b2_h(0) = 0.0f; b2_h(1) = -1.0f;
      Kokkos::deep_copy(b1, b1_h);
      Kokkos::deep_copy(b2, b2_h);

      auto mixed_model = ponni::create_inference_model(
          ponni::Matvec<float>(w1), ponni::Bias<float>(b1), ponni::Relu<float>(3),
          ponni::Softmax<float>(3),
          ponni::Matvec<float>(w2), ponni::Bias<float>(b2), ponni::Tanh<float>(2));
      using MixedModel = decltype(mixed_model);
      static_assert(MixedModel::get_num_fused_dense_blocks() == 2,
                    "The planner should discover fused blocks on both sides of a barrier");

      real2d mixed_input("mixed_input", 2, 2);
      Kokkos::deep_copy(mixed_input, 0.5f);
      auto mixed_output = mixed_model.forward_batch_parallel(mixed_input);
      auto mixed_output_h = ponni::create_host_copy(mixed_output);
      real2d mixed_output_in_kernel("mixed_output_in_kernel", 2, 2);
      auto const mixed_params = mixed_model.params;
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), 2, KOKKOS_LAMBDA(int ibatch) {
        MixedModel::forward_batch_parallel_in_kernel(mixed_input, mixed_output_in_kernel,
                                                     mixed_params, ibatch);
      });
      auto mixed_output_in_kernel_h = ponni::create_host_copy(mixed_output_in_kernel);
      require_true(mixed_model.params.tmp1.extent(0) == 3 &&
                   mixed_model.params.tmp1.extent(1) == 2,
                   "Mixed planning should materialize the pre-barrier state in tmp1");
      require_true(!mixed_model.params.tmp2.is_allocated(),
                   "A final fused block should not allocate a second temporary View");
      for (int ibatch = 0; ibatch < 2; ibatch++) {
        require_true(nearly_equal(mixed_output_h(0,ibatch), std::tanh(1.0f), 1e-5f),
                     "Mixed fused/barrier output 0 is incorrect");
        require_true(nearly_equal(mixed_output_h(1,ibatch), 0.0f, 1e-5f),
                     "Mixed fused/barrier output 1 is incorrect");
        require_true(nearly_equal(mixed_output_in_kernel_h(0,ibatch), std::tanh(1.0f), 1e-5f) &&
                     nearly_equal(mixed_output_in_kernel_h(1,ibatch), 0.0f, 1e-5f),
                     "Intra-kernel mixed fused/barrier output is incorrect");
      }
    }

    // The factory is the only place where users select execution and memory.
    // It rebinds all supplied layers, including their learned parameters.
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostMemorySpace = Kokkos::HostSpace;
    auto host_model = ponni::create_inference_model(
        HostExecutionSpace(),
        HostMemorySpace(),
        ponni::Matvec<float>(weights),
        ponni::Bias<float>(bias),
        ponni::LeakyRelu<float>(2, 0.1f));
    static_assert(std::is_same_v<typename decltype(host_model)::execution_space,HostExecutionSpace>);
    static_assert(std::is_same_v<typename decltype(host_model)::memory_space,HostMemorySpace>);
    static_assert(std::is_same_v<
        typename std::tuple_element_t<0,decltype(host_model.params.layers)>::memory_space,
        HostMemorySpace>);

    // Public Views retain the LayoutRight contract even when inference is
    // rebound to a host execution and memory space.
    Kokkos::View<float**, Kokkos::LayoutRight, HostMemorySpace> host_input("host_input", 2, 1);
    Kokkos::View<float**, Kokkos::LayoutRight, HostMemorySpace> host_output("host_output", 2, 1);
    host_input(0,0) = 1.0f;
    host_input(1,0) = 2.0f;
    host_model.forward_batch_parallel(host_input, host_output);
    require_true(nearly_equal(host_output(0,0), 4.0f) && nearly_equal(host_output(1,0), 4.0f),
                 "Custom host execution/memory model produced incorrect output");

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
      concat_layer.validate(2);

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
      save_layer.validate();
    }

    // Cover layer-level API options, including trainable flags and parameter updates.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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

      mv.validate();

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

      bias.validate();

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
      add_layer.validate(3);

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
      proj_nt.validate(2);
    }

    // Cover every activation's host API plus its SArray and default Kokkos memory-space compute paths.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;

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

      ln.validate();

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

    // Cover the MinMaxNorm forward paths.
    {
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
      ponni::MinMaxNorm<float> mm(3, -1.0f, 1.0f);
      require_true(std::string(mm.get_label()) == "MinMaxNorm", "MinMaxNorm label is incorrect");
      require_true(mm.get_num_inputs() == 3 && mm.get_num_outputs() == 3 &&
                   ponni::MinMaxNorm<float>::get_num_inputs(mm.params) == 3 &&
                   ponni::MinMaxNorm<float>::get_num_outputs(mm.params) == 3,
                   "MinMaxNorm input/output size is incorrect");
      Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space> no_mm_parameters;
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

      mm.validate();
    }

    // Cover the projection skip layer, including trainable parameters.
    {
      using real1d = Kokkos::View<float*, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight, typename Kokkos::DefaultExecutionSpace::memory_space>;
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

      proj.validate(2);

      Proj initialized_proj(3, 2, true, ponni::Initializer_Constant<float>(0.25f));
      initialized_proj.validate(2);
      auto initialized_proj_weights_h = ponni::create_host_copy(initialized_proj.params.weights);
      auto initialized_proj_bias_h = ponni::create_host_copy(initialized_proj.params.bias);
      require_true(nearly_equal(initialized_proj_weights_h(1,2), 0.25f) &&
                   nearly_equal(initialized_proj_bias_h(2), 0.25f),
                   "Projection skip initializer constructor failed");
    }

    // Exercise the initializer formulas on a large enough sample that the
    // statistical checks are stable while remaining inexpensive on a GPU.
    {
      using real2d = Kokkos::View<float**, Kokkos::LayoutRight,
                                  typename Kokkos::DefaultExecutionSpace::memory_space>;
      int constexpr fan_in = 512;
      int constexpr fan_out = 384;
      real2d values("initializer_values", fan_in, fan_out);
      real2d repeated("initializer_repeated", fan_in, fan_out);

      auto check_variance = [&](SampleStats const & stats, double expected, std::string const & name) {
        double const relative_error = std::abs(stats.variance - expected) / expected;
        require_true(relative_error < 0.05, name + " variance does not match its documented formula");
      };

      auto fill_twice = [&]<class Initializer>(Initializer const & initializer, std::string const & name) {
        initializer.fill(values);
        initializer.fill(repeated);
        require_true(exactly_equal(values,repeated), name + " is not deterministic for a nonzero seed");
        return sample_stats(values);
      };

      ponni::Initializer_Zeros<float>().fill(values);
      auto stats = sample_stats(values);
      require_true(stats.minimum == 0.0 && stats.maximum == 0.0, "Initializer_Zeros failed");

      ponni::Initializer_Ones<float>().fill(values);
      stats = sample_stats(values);
      require_true(stats.minimum == 1.0 && stats.maximum == 1.0, "Initializer_Ones failed");

      ponni::Initializer_Constant<float>(2.5f).fill(values);
      stats = sample_stats(values);
      require_true(stats.minimum == 2.5 && stats.maximum == 2.5, "Initializer_Constant failed");

      stats = fill_twice(ponni::Initializer_Random_Uniform<float>(-0.2f,0.3f,111), "Random uniform");
      require_true(stats.minimum >= -0.2 && stats.maximum < 0.3,
                   "Initializer_Random_Uniform violated its complete bounds");
      require_true(std::abs(stats.mean - 0.05) < 0.005,
                   "Initializer_Random_Uniform mean does not match its interval");
      check_variance(stats, 0.5 * 0.5 / 12.0, "Initializer_Random_Uniform");

      stats = fill_twice(ponni::Initializer_Random_Normal<float>(0.35f,0.4f,222), "Random normal");
      require_true(std::abs(stats.mean - 0.35) < 0.01,
                   "Initializer_Random_Normal mean does not match its parameter");
      check_variance(stats, 0.4 * 0.4, "Initializer_Random_Normal");

      stats = fill_twice(ponni::Initializer_Truncated_Normal<float>(-0.2f,0.3f,333), "Truncated normal");
      require_true(stats.minimum >= -0.8 && stats.maximum <= 0.4,
                   "Initializer_Truncated_Normal violated its two-sigma bounds");
      require_true(std::abs(stats.mean + 0.2) < 0.01,
                   "Initializer_Truncated_Normal mean does not match its parameter");
      // A standard normal truncated symmetrically at two sigma has variance
      // 1 - 4*phi(2)/(2*Phi(2)-1), approximately 0.7737413.
      check_variance(stats, 0.3 * 0.3 * 0.7737413, "Initializer_Truncated_Normal");

      double const xavier_variance = 2.0 / static_cast<double>(fan_in + fan_out);
      double const he_variance = 2.0 / static_cast<double>(fan_in);
      double const lecun_variance = 1.0 / static_cast<double>(fan_in);

      stats = fill_twice(ponni::Initializer_Xavier_Uniform<float>(444), "Xavier uniform");
      double const xavier_limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
      require_true(stats.minimum >= -xavier_limit && stats.maximum < xavier_limit,
                   "Initializer_Xavier_Uniform violated its calculated bounds");
      check_variance(stats, xavier_variance, "Initializer_Xavier_Uniform");

      stats = fill_twice(ponni::Initializer_Xavier_Normal<float>(555), "Xavier normal");
      check_variance(stats, xavier_variance, "Initializer_Xavier_Normal");

      stats = fill_twice(ponni::Initializer_He_Uniform<float>(666), "He uniform");
      double const he_limit = std::sqrt(6.0 / static_cast<double>(fan_in));
      require_true(stats.minimum >= -he_limit && stats.maximum < he_limit,
                   "Initializer_He_Uniform violated its calculated bounds");
      check_variance(stats, he_variance, "Initializer_He_Uniform");

      stats = fill_twice(ponni::Initializer_He_Normal<float>(777), "He normal");
      check_variance(stats, he_variance, "Initializer_He_Normal");

      stats = fill_twice(ponni::Initializer_Lecun_Uniform<float>(888), "Lecun uniform");
      double const lecun_limit = std::sqrt(3.0 / static_cast<double>(fan_in));
      require_true(stats.minimum >= -lecun_limit && stats.maximum < lecun_limit,
                   "Initializer_Lecun_Uniform violated its calculated bounds");
      check_variance(stats, lecun_variance, "Initializer_Lecun_Uniform");

      stats = fill_twice(ponni::Initializer_Lecun_Normal<float>(999), "Lecun normal");
      check_variance(stats, lecun_variance, "Initializer_Lecun_Normal");

      // Zero-length Views are valid no-op targets for every initializer,
      // including fan-based formulas that would otherwise divide by zero.
      real2d empty("initializer_empty", 0, fan_out);
      ponni::Initializer_None<float>().fill(empty);
      ponni::Initializer_Zeros<float>().fill(empty);
      ponni::Initializer_Ones<float>().fill(empty);
      ponni::Initializer_Constant<float>(2.5f).fill(empty);
      ponni::Initializer_Random_Uniform<float>(-1.0f,1.0f,1).fill(empty);
      ponni::Initializer_Random_Normal<float>(0.0f,1.0f,2).fill(empty);
      ponni::Initializer_Truncated_Normal<float>(0.0f,1.0f,3).fill(empty);
      ponni::Initializer_Xavier_Uniform<float>(4).fill(empty);
      ponni::Initializer_Xavier_Normal<float>(5).fill(empty);
      ponni::Initializer_He_Uniform<float>(6).fill(empty);
      ponni::Initializer_He_Normal<float>(7).fill(empty);
      ponni::Initializer_Lecun_Uniform<float>(8).fill(empty);
      ponni::Initializer_Lecun_Normal<float>(9).fill(empty);
      ponni::Initializer_Orthogonal<float>(1.0f,10).fill(empty);
      require_true(empty.size() == 0, "Initializers changed a zero-sized View");

      // Degenerate, but valid, distributions should fill their exact value.
      ponni::Initializer_Random_Uniform<float>(0.75f,0.75f,11).fill(values);
      stats = sample_stats(values);
      require_true(stats.minimum == 0.75 && stats.maximum == 0.75,
                   "Equal Random_Uniform bounds should produce a constant");
      ponni::Initializer_Random_Normal<float>(-0.5f,0.0f,12).fill(values);
      stats = sample_stats(values);
      require_true(stats.minimum == -0.5 && stats.maximum == -0.5,
                   "Zero Random_Normal deviation should produce its mean");
      ponni::Initializer_Truncated_Normal<float>(0.25f,0.0f,13).fill(values);
      stats = sample_stats(values);
      require_true(stats.minimum == 0.25 && stats.maximum == 0.25,
                   "Zero Truncated_Normal deviation should produce its mean");

      float const infinity = std::numeric_limits<float>::infinity();
      float const nan = std::numeric_limits<float>::quiet_NaN();
      require_true(throws_invalid_argument([] { ponni::Initializer_Random_Uniform<float>(1.0f,-1.0f,1); }),
                   "Random_Uniform should reject reversed bounds");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Random_Uniform<float>(nan,1.0f,1); }),
                   "Random_Uniform should reject a non-finite lower bound");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Random_Uniform<float>(0.0f,infinity,1); }),
                   "Random_Uniform should reject a non-finite upper bound");
      require_true(throws_invalid_argument([] { ponni::Initializer_Random_Normal<float>(0.0f,-1.0f,1); }),
                   "Random_Normal should reject a negative standard deviation");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Random_Normal<float>(nan,1.0f,1); }),
                   "Random_Normal should reject a non-finite mean");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Random_Normal<float>(0.0f,infinity,1); }),
                   "Random_Normal should reject a non-finite standard deviation");
      require_true(throws_invalid_argument([] { ponni::Initializer_Truncated_Normal<float>(0.0f,-1.0f,1); }),
                   "Truncated_Normal should reject a negative standard deviation");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Truncated_Normal<float>(nan,1.0f,1); }),
                   "Truncated_Normal should reject a non-finite mean");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Truncated_Normal<float>(0.0f,infinity,1); }),
                   "Truncated_Normal should reject a non-finite standard deviation");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Orthogonal<float>(infinity,1); }),
                   "Orthogonal should reject a non-finite gain");
      require_true(throws_invalid_argument([&] { ponni::Initializer_Orthogonal<float>(nan,1); }),
                   "Orthogonal should reject a NaN gain");
    }

    // Exercise both lanes and the packed multiply-add on the active device backend.
    {
      Kokkos::View<float*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
          result("two_half_result", 16);
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
      require_true(is_ieee_nan(result_h(8)) && result_h(9) == 0.0f && !std::signbit(result_h(9)),
                   "TwoHalf ONNX sign handling produced incorrect lanes: low=" +
                   std::to_string(result_h(8)) + ", high=" + std::to_string(result_h(9)) +
                   ", high_signbit=" + std::to_string(std::signbit(result_h(9))));
      require_true(nearly_equal(result_h(10), 0.0f) && nearly_equal(result_h(11), 0.0f) &&
                   nearly_equal(result_h(12), 0.0f) && nearly_equal(result_h(13), 1.0f) &&
                   nearly_equal(result_h(14), 4.0f) && nearly_equal(result_h(15), -3.0f),
                   "TwoMask comparison, logical, and selection operations produced incorrect lanes");
    }
  }

  Kokkos::finalize();

  if (!ok) return 1;
  std::cout << "core_unit passed" << std::endl;
  return 0;
}
