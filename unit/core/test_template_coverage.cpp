#include "ponni.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {


struct Check {
  bool passed = true;

  void operator()(bool condition, std::string const & message) {
    if (condition) return;
    passed = false;
    std::cerr << "FAILED: " << message << std::endl;
  }
};

template <class Scalar>
bool close(Scalar value, double expected, double tolerance = 1.e-6) {
  return std::abs(static_cast<double>(value) - expected) <= tolerance;
}

std::vector<unsigned char> read_bytes(std::string const & path) {
  std::ifstream input(path,std::ios::binary | std::ios::ate);
  std::streamsize const size = input.tellg();
  input.seekg(0,std::ios::beg);
  std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()),size);
  return bytes;
}

void write_bytes(std::string const & path, std::vector<unsigned char> const & bytes) {
  std::ofstream output(path,std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<char const *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
}

bool replace_bytes(std::vector<unsigned char> & bytes, std::string const & before, std::string const & after) {
  if (before.size() != after.size()) return false;
  auto const found = std::search(bytes.begin(),bytes.end(),before.begin(),before.end());
  if (found == bytes.end()) return false;
  std::copy(after.begin(),after.end(),found);
  return true;
}

std::vector<unsigned char> make_safetensors_bytes(std::string header, std::size_t payload_size = 0) {
  header.append((8 - header.size() % 8) % 8,' ');
  std::uint64_t const header_size = header.size();
  std::vector<unsigned char> bytes(8);
  for (int byte = 0; byte < 8; byte++) {
    bytes[byte] = static_cast<unsigned char>((header_size >> (8 * byte)) & 0xffu);
  }
  bytes.insert(bytes.end(),header.begin(),header.end());
  bytes.resize(bytes.size() + payload_size,0);
  return bytes;
}

template <class Scalar>
Kokkos::View<Scalar**,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
make_matrix(std::string const & label, int rows, int columns, std::initializer_list<double> values) {
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
  using View = Kokkos::View<Scalar**,Kokkos::LayoutRight,MemorySpace>;
  View result(label,rows,columns);
  auto host = Kokkos::create_mirror_view(result);
  auto value = values.begin();
  for (int row = 0; row < rows; row++) {
    for (int column = 0; column < columns; column++) {
      host(row,column) = static_cast<Scalar>(*value++);
    }
  }
  Kokkos::deep_copy(result,host);
  return result;
}

// Test-only dense layer demonstrating the custom dense contract. Its fixed
// 2x2 transform is intentionally non-diagonal so a faulty elementwise
// implementation cannot accidentally produce the expected answer.
template <class Real = float, int N = 2,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct TestDense {
  using memory_space = MemorySpace;
  template <class NewMemorySpace> using rebind_memory_space = TestDense<Real,N,NewMemorySpace>;
  using real1d = Kokkos::View<Real*,Kokkos::LayoutRight,MemorySpace>;

  bool static constexpr overwrite_input = false;
  bool static constexpr binop = false;
  bool static constexpr save = false;
  ponni::LayerFusionKind static constexpr fusion_kind = ponni::LayerFusionKind::dense;
  int static constexpr INPUT_SIZE = N;
  int static constexpr OUTPUT_SIZE = N;

  struct Params {
    int num_inputs = N;
    Real m00 = static_cast<Real>(2);
    Real m01 = static_cast<Real>(1);
    Real m10 = static_cast<Real>(1);
    Real m11 = static_cast<Real>(-1);
  };
  Params params;

  char const * get_label() const { return "TestDense"; }
  KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
  KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
  int get_num_inputs() const { return params.num_inputs; }
  int get_num_outputs() const { return params.num_inputs; }
  int get_num_trainable_parameters() const { return 0; }

  template <class NewMemorySpace>
  auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
    TestDense<Real,N,NewMemorySpace> result;
    result.params = {params.num_inputs,params.m00,params.m01,params.m10,params.m11};
    return result;
  }

  template <class InputView>
  KOKKOS_INLINE_FUNCTION static Real compute_output(InputView const & input, int feature, int batch,
                                                     Params const & p) {
    return feature == 0 ? p.m00 * input(0,batch) + p.m01 * input(1,batch)
                        : p.m10 * input(0,batch) + p.m11 * input(1,batch);
  }

  template <class InputView, class OutputView>
  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                          int batch, Params const & p) {
    for (int feature = 0; feature < N; feature++) output(feature,batch) = compute_output(input,feature,batch,p);
  }

  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<Real,N> const & input,
                                                          ponni::SArray<Real,N> & output, Params const & p) {
    output(0) = p.m00 * input(0) + p.m01 * input(1);
    output(1) = p.m10 * input(0) + p.m11 * input(1);
  }

  void set_trainable_parameters(real1d const &) {}
  real1d get_trainable_parameters() const { return real1d(); }

  void validate() const {
    if (params.num_inputs != N) Kokkos::abort("TestDense requires its declared fixed size");
  }
};

// Test-only indexed pointwise layer. The feature-dependent term forces the
// fusion dispatcher to use apply_fused(value, feature, params).
template <class Real = float, int N = 2,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct TestIndexedPointwise {
  using memory_space = MemorySpace;
  template <class NewMemorySpace> using rebind_memory_space = TestIndexedPointwise<Real,N,NewMemorySpace>;
  using real1d = Kokkos::View<Real*,Kokkos::LayoutRight,MemorySpace>;

  bool static constexpr overwrite_input = true;
  bool static constexpr binop = false;
  bool static constexpr save = false;
  ponni::LayerFusionKind static constexpr fusion_kind = ponni::LayerFusionKind::pointwise;
  int static constexpr INPUT_SIZE = N;
  int static constexpr OUTPUT_SIZE = N;

  struct Params {
    int num_inputs = N;
    Real offset = static_cast<Real>(0.5);
  };
  Params params;

  char const * get_label() const { return "TestIndexedPointwise"; }
  KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
  KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
  int get_num_inputs() const { return params.num_inputs; }
  int get_num_outputs() const { return params.num_inputs; }
  int get_num_trainable_parameters() const { return 0; }

  template <class NewMemorySpace>
  auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
    TestIndexedPointwise<Real,N,NewMemorySpace> result;
    result.params = {params.num_inputs,params.offset};
    return result;
  }

  KOKKOS_INLINE_FUNCTION static Real apply_fused(Real value, int feature, Params const & p) {
    return value + p.offset * static_cast<Real>(feature + 1);
  }

  template <class InputView, class OutputView>
  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                          int batch, Params const & p) {
    for (int feature = 0; feature < N; feature++) {
      output(feature,batch) = apply_fused(input(feature,batch),feature,p);
    }
  }

  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<Real,N> const & input,
                                                          ponni::SArray<Real,N> & output, Params const & p) {
    for (int feature = 0; feature < N; feature++) output(feature) = apply_fused(input(feature),feature,p);
  }

  void set_trainable_parameters(real1d const &) {}
  real1d get_trainable_parameters() const { return real1d(); }

  void validate() const {
    if (params.num_inputs != N) Kokkos::abort("TestIndexedPointwise requires its declared fixed size");
  }
};

// No fusion_kind is declared: the default LayerTraits specialization must
// conservatively materialize this feature-swapping operation as a barrier.
template <class Real = float, int N = 2,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct TestBarrier {
  using memory_space = MemorySpace;
  template <class NewMemorySpace> using rebind_memory_space = TestBarrier<Real,N,NewMemorySpace>;
  using real1d = Kokkos::View<Real*,Kokkos::LayoutRight,MemorySpace>;

  bool static constexpr overwrite_input = false;
  bool static constexpr binop = false;
  bool static constexpr save = false;
  int static constexpr INPUT_SIZE = N;
  int static constexpr OUTPUT_SIZE = N;

  struct Params {
    int num_inputs = N;
    Real scale = static_cast<Real>(1);
  };
  Params params;

  char const * get_label() const { return "TestBarrier"; }
  KOKKOS_INLINE_FUNCTION static int get_num_inputs(Params const & p) { return p.num_inputs; }
  KOKKOS_INLINE_FUNCTION static int get_num_outputs(Params const & p) { return p.num_inputs; }
  int get_num_inputs() const { return params.num_inputs; }
  int get_num_outputs() const { return params.num_inputs; }
  int get_num_trainable_parameters() const { return 0; }

  template <class NewMemorySpace>
  auto copy_to_memory_space(NewMemorySpace const & = NewMemorySpace()) const {
    TestBarrier<Real,N,NewMemorySpace> result;
    result.params = {params.num_inputs,params.scale};
    return result;
  }

  template <class InputView, class OutputView>
  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(InputView const & input, OutputView const & output,
                                                          int batch, Params const & p) {
    output(0,batch) = p.scale * input(1,batch);
    output(1,batch) = p.scale * input(0,batch);
  }

  KOKKOS_INLINE_FUNCTION static void compute_all_outputs(ponni::SArray<Real,N> const & input,
                                                          ponni::SArray<Real,N> & output, Params const & p) {
    output(0) = p.scale * input(1);
    output(1) = p.scale * input(0);
  }

  void set_trainable_parameters(real1d const &) {}
  real1d get_trainable_parameters() const { return real1d(); }

  void validate() const {
    if (params.num_inputs != N) Kokkos::abort("TestBarrier requires its declared fixed size");
  }
};

template <class Model>
void check_precision_round_trip(Model & model, std::string const & path, double tolerance, Check & check) {
  using Real = typename Model::real1d::non_const_value_type;
  typename Model::real2d input("precision_input",2,1);
  auto input_host = Kokkos::create_mirror_view(input);
  input_host(0,0) = static_cast<Real>(3);
  input_host(1,0) = static_cast<Real>(1);
  Kokkos::deep_copy(input,input_host);

  auto output_host = ponni::create_host_copy(model.forward_batch_parallel(input));
  check(close(output_host(0,0),5.0,tolerance),path + " initial inference is incorrect");

  std::string error;
  check(model.save_weights(path,&error),path + " save failed: " + error);
  typename Model::real1d zero_parameters("precision_zero_parameters",model.get_num_trainable_parameters());
  Kokkos::deep_copy(zero_parameters,static_cast<Real>(0));
  model.set_trainable_parameters(zero_parameters);
  output_host = ponni::create_host_copy(model.forward_batch_parallel(input));
  check(close(output_host(0,0),0.0,tolerance),path + " parameter replacement is incorrect");

  error.clear();
  check(model.load_weights(path,&error),path + " load failed: " + error);
  output_host = ponni::create_host_copy(model.forward_batch_parallel(input));
  check(close(output_host(0,0),5.0,tolerance),path + " round-trip inference is incorrect");
  std::remove(path.c_str());
}

void test_sarray_api(Check & check) {
  using Array = ponni::SArray<int,2,3>;
  using RightView = Kokkos::View<float**,Kokkos::LayoutRight,Kokkos::HostSpace>;
  using LeftView = Kokkos::View<float**,Kokkos::LayoutLeft,Kokkos::HostSpace>;
  static_assert(ponni::is_layout_right_view_v<RightView>);
  static_assert(!ponni::is_layout_right_view_v<LeftView>);
  static_assert(Array::is_SArray);
  static_assert(Array::rank == 2);
  static_assert(Array::num_elements == 6);
  static_assert(Array::size() == 6);
  static_assert(Array::template extent<0>() == 2);
  static_assert(Array::template extent<1>() == 3);
  static_assert(std::is_same_v<Array::value_type,int>);
  static_assert(std::is_same_v<Array::const_value_type,int const>);
  static_assert(std::is_same_v<Array::non_const_value_type,int>);

  Kokkos::View<int*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
      results("sarray_results",16);
  Kokkos::parallel_for("SArray complete API",1,KOKKOS_LAMBDA(int) {
    Array values;
    values = 3.5;
    values(1,2) = 11;
    values.data()[1] = 5;
    values.my_data[2] = 7;
    int sum = 0;
    for (int const * value = values.begin(); value != values.end(); value++) sum += *value;
    auto const dimensions = values.extents();
    results(0) = values(0,0);
    results(1) = values(0,1);
    results(2) = values(0,2);
    results(3) = values(1,2);
    results(4) = sum;
    results(5) = static_cast<int>(values.size());
    results(6) = static_cast<int>(values.extent(0));
    results(7) = static_cast<int>(values.extent(1));
    results(8) = static_cast<int>(dimensions(0));
    results(9) = static_cast<int>(dimensions(1));
    results(10) = values.begin() == values.data() ? 1 : 0;
    results(11) = values.end() - values.begin();
    ponni::SArray<int,2,3,4> cube;
    cube = 0;
    cube(1,2,3) = 17;
    auto const cube_dimensions = cube.extents();
    results(12) = cube.data()[23];
    results(13) = static_cast<int>(cube_dimensions(0));
    results(14) = static_cast<int>(cube_dimensions(1));
    results(15) = static_cast<int>(cube_dimensions(2));
  });
  auto host = ponni::create_host_copy(results);
  check(host(0) == 3 && host(1) == 5 && host(2) == 7 && host(3) == 11,
        "SArray indexing, scalar assignment, data(), or public storage is incorrect");
  check(host(4) == 32 && host(5) == 6 && host(6) == 2 && host(7) == 3,
        "SArray iterators, size(), or runtime extent() is incorrect");
  check(host(8) == 2 && host(9) == 3 && host(10) == 1 && host(11) == 6,
        "SArray extents() or pointer range is incorrect");
  check(host(12) == 17 && host(13) == 2 && host(14) == 3 && host(15) == 4,
        "SArray rank-three indexing or extents() is incorrect");

  Array printable;
  printable = 4;
  std::ostringstream stream;
  stream << printable;
  check(stream.str() == "ponni::SArray: 4 , 4 , 4 , 4 , 4 , 4\n", "SArray stream output is incorrect");
}

void test_builtin_sarray_and_single_layer(Check & check) {
  auto first_weights = make_matrix<float>("sarray_first_weights",2,2,{1.0,2.0,3.0,4.0});
  auto second_weights = make_matrix<float>("sarray_second_weights",2,1,{2.0,-1.0});
  Kokkos::View<float*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
      bias("sarray_bias",2);
  auto bias_host = Kokkos::create_mirror_view(bias);
  bias_host(0) = 0.5f;
  bias_host(1) = -1.0f;
  Kokkos::deep_copy(bias,bias_host);

  auto model = ponni::create_inference_model(
      ponni::Matvec<float,2,2>(first_weights), ponni::Bias<float,2>(bias), ponni::Relu<float,2>(2),
      ponni::Matvec<float,2,1>(second_weights));
  using Model = decltype(model);
  Kokkos::View<float*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
      result("sarray_model_result",1);
  auto const params = model.params;
  Kokkos::parallel_for("Complete template SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> input;
    ponni::SArray<float,1> output;
    input(0) = 1.0f;
    input(1) = 2.0f;
    Model::forward_batch_parallel_in_kernel(input,output,params);
    result(0) = output(0);
  });
  check(close(ponni::create_host_copy(result)(0),6.0),"Complete built-in SArray model produced an incorrect answer");

  auto single_layer = ponni::create_inference_model(ponni::Relu<float,2>(2));
  using SingleLayer = decltype(single_layer);
  typename SingleLayer::real2d input("single_layer_input",2,1);
  auto input_host = Kokkos::create_mirror_view(input);
  input_host(0,0) = -2.0f;
  input_host(1,0) = 3.0f;
  Kokkos::deep_copy(input,input_host);
  auto output_host = ponni::create_host_copy(single_layer.forward_batch_parallel(input));
  check(close(output_host(0,0),0.0) && close(output_host(1,0),3.0),
        "Single-layer View model produced an incorrect answer");

  Kokkos::View<float*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
      single_result("single_sarray_result",2);
  auto const single_params = single_layer.params;
  Kokkos::parallel_for("Single layer SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> local_input;
    ponni::SArray<float,2> local_output;
    local_input(0) = -2.0f;
    local_input(1) = 3.0f;
    SingleLayer::forward_batch_parallel_in_kernel(local_input,local_output,single_params);
    single_result(0) = local_output(0);
    single_result(1) = local_output(1);
  });
  auto single_host = ponni::create_host_copy(single_result);
  check(close(single_host(0),0.0) && close(single_host(1),3.0),
        "Single-layer SArray model produced an incorrect answer");
}

void test_saved_state_model_edges(Check & check) {
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;

  // Saving the raw model input is a useful residual pattern and specifically
  // guards the leading-barrier path, which must materialize both the saved
  // branch and the main branch before continuing.
  auto leading_save = ponni::create_inference_model(
      ponni::Save_State<0,float,2>(2), ponni::Relu<float,2>(2), ponni::Binop_Add<0,float,2>(2));
  using LeadingSaveModel = decltype(leading_save);
  typename LeadingSaveModel::real2d input("leading_save_input",2,1);
  auto input_host = Kokkos::create_mirror_view(input);
  input_host(0,0) = -2.0f;
  input_host(1,0) =  3.0f;
  Kokkos::deep_copy(input,input_host);
  auto output_host = ponni::create_host_copy(leading_save.forward_batch_parallel(input));
  check(close(output_host(0,0),-2.0) && close(output_host(1,0),6.0),
        "A leading Save_State did not preserve both residual branches");

  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> leading_sarray_result("leading_sarray_result",2);
  auto const leading_params = leading_save.params;
  Kokkos::parallel_for("Leading Save_State SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> local_input;
    ponni::SArray<float,2> local_output;
    local_input(0) = -2.0f;
    local_input(1) =  3.0f;
    LeadingSaveModel::forward_batch_parallel_in_kernel(local_input,local_output,leading_params);
    leading_sarray_result(0) = local_output(0);
    leading_sarray_result(1) = local_output(1);
  });
  auto leading_sarray_host = ponni::create_host_copy(leading_sarray_result);
  check(close(leading_sarray_host(0),-2.0) && close(leading_sarray_host(1),6.0),
        "SArray Save_State/Binop_Add traversal produced an incorrect answer");

  auto projection_weights = make_matrix<float>("sarray_projection_weights",2,2,{1.0,0.0,0.0,1.0});
  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> projection_bias("sarray_projection_bias",2);
  Kokkos::deep_copy(projection_bias,0.0f);
  auto projection = ponni::create_inference_model(
      ponni::Save_State<0,float,2>(2), ponni::Relu<float,2>(2),
      ponni::Binop_Projection_Add<0,float,2,2>(projection_weights,projection_bias,false));
  using ProjectionModel = decltype(projection);
  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> projection_result("projection_sarray_result",2);
  auto const projection_params = projection.params;
  Kokkos::parallel_for("Projection residual SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> local_input;
    ponni::SArray<float,2> local_output;
    local_input(0) = -2.0f;
    local_input(1) =  3.0f;
    ProjectionModel::forward_batch_parallel_in_kernel(local_input,local_output,projection_params);
    projection_result(0) = local_output(0);
    projection_result(1) = local_output(1);
  });
  auto projection_host = ponni::create_host_copy(projection_result);
  check(close(projection_host(0),-2.0) && close(projection_host(1),6.0),
        "SArray Save_State/Binop_Projection_Add traversal produced an incorrect answer");

  // Keep differently sized residuals live simultaneously. Slot zero remains
  // live while slot one is saved and consumed, then slot zero is consumed by
  // the final concatenation.
  auto multi_save = ponni::create_inference_model(
      ponni::Save_State<0,float,2>(2), ponni::Relu<float,2>(2),
      ponni::Binop_Concatenate<0,float,2,2>(2,4,true), ponni::Save_State<1,float,4>(4),
      ponni::Relu<float,4>(4), ponni::Binop_Add<1,float,4>(4),
      ponni::Binop_Concatenate<0,float,4,2>(4,6,true));
  using MultiSaveModel = decltype(multi_save);
  static_assert(MultiSaveModel::get_num_saved_states() == 2);
  auto multi_output_host = ponni::create_host_copy(multi_save.forward_batch_parallel(input));
  std::array<double,6> const expected{0.0,6.0,-2.0,6.0,-2.0,3.0};
  for (int i = 0; i < 6; i++) {
    check(close(multi_output_host(i,0),expected[i]),
          "Multiple View saved-state slots produced an incorrect answer at feature " + std::to_string(i));
  }

  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> multi_result("multi_save_sarray_result",6);
  auto const multi_params = multi_save.params;
  Kokkos::parallel_for("Multiple saved-state SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> local_input;
    ponni::SArray<float,6> local_output;
    local_input(0) = -2.0f;
    local_input(1) =  3.0f;
    MultiSaveModel::forward_batch_parallel_in_kernel(local_input,local_output,multi_params);
    for (int i = 0; i < 6; i++) multi_result(i) = local_output(i);
  });
  auto multi_result_host = ponni::create_host_copy(multi_result);
  for (int i = 0; i < 6; i++) {
    check(close(multi_result_host(i),expected[i]),
          "Multiple SArray saved-state slots produced an incorrect answer at feature " + std::to_string(i));
  }
}

void test_precision_factories(Check & check) {
  auto float_weights = make_matrix<float>("float_factory_weights",2,1,{2.0,-1.0});
  auto float_model = ponni::create_inference_model_single_precision(ponni::Matvec<float,2,1>(float_weights));
  static_assert(std::is_same_v<typename decltype(float_model)::real1d::non_const_value_type,float>);
  check_precision_round_trip(float_model,"template_float.ponni",1.e-6,check);

  auto double_weights = make_matrix<double>("double_factory_weights",2,1,{2.0,-1.0});
  auto double_model = ponni::create_inference_model_double_precision(ponni::Matvec<double,2,1>(double_weights));
  static_assert(std::is_same_v<typename decltype(double_model)::real1d::non_const_value_type,double>);
  check_precision_round_trip(double_model,"template_double.ponni",1.e-12,check);

  using Half = Kokkos::Experimental::half_t;
  auto half_weights = make_matrix<Half>("half_factory_weights",2,1,{2.0,-1.0});
  auto half_model = ponni::create_inference_model_half_precision(ponni::Matvec<Half,2,1>(half_weights));
  static_assert(std::is_same_v<typename decltype(half_model)::real1d::non_const_value_type,Half>);
  check_precision_round_trip(half_model,"template_half.ponni",2.e-2,check);

  using BHalf = Kokkos::Experimental::bhalf_t;
  auto bhalf_weights = make_matrix<BHalf>("bhalf_factory_weights",2,1,{2.0,-1.0});
  auto bhalf_model = ponni::create_inference_model_bhalf_precision(ponni::Matvec<BHalf,2,1>(bhalf_weights));
  static_assert(std::is_same_v<typename decltype(bhalf_model)::real1d::non_const_value_type,BHalf>);
  check_precision_round_trip(bhalf_model,"template_bhalf.ponni",5.e-2,check);
}

void test_ponni_tensor_ranks(Check & check) {
  std::vector<double> values(14);
  for (int i = 0; i < static_cast<int>(values.size()); i++) values[i] = 0.25 + static_cast<double>(i);
  std::vector<ponni::PonniTensorSpec> const specs{
      {"rank1","F64",{2},0},
      {"rank2","F64",{2,2},2},
      {"rank3","F64",{2,1,2},6},
      {"rank4","F64",{1,2,1,2},10},
  };
  std::string const path = "template_tensor_ranks.ponni";
  std::string error;
  check(ponni::write_ponni_file(path,specs,"template-rank-test",values.data(),&error,"test"),
        "Writing rank-test PONNI file failed: " + error);

  ponni::PonniFile file;
  check(file.load(path,&error),"Loading rank-test PONNI file failed: " + error);
  auto rank1 = ponni::load_ponni_tensor<1>(file,"rank1");
  auto rank2 = ponni::load_ponni_tensor<2,Kokkos::HostSpace>(file,"rank2");
  auto rank3 = ponni::load_ponni_tensor<3>(file,"rank3");
  auto rank4 = ponni::load_ponni_tensor<4,Kokkos::HostSpace>(path,"rank4");
  auto rank1_host = ponni::create_host_copy(rank1);
  auto rank3_host = ponni::create_host_copy(rank3);
  check(rank1.extent(0) == 2 && close(rank1_host(1),1.25),"Rank-one F64 PONNI tensor load is incorrect");
  check(rank2.extent(0) == 2 && rank2.extent(1) == 2 && close(rank2(1,1),5.25),
        "Rank-two explicit-memory PONNI tensor load is incorrect");
  check(rank3.extent(0) == 2 && rank3.extent(1) == 1 && rank3.extent(2) == 2 && close(rank3_host(1,0,1),9.25),
        "Rank-three PONNI tensor load is incorrect");
  check(rank4.extent(0) == 1 && rank4.extent(1) == 2 && rank4.extent(2) == 1 && rank4.extent(3) == 2 &&
        close(rank4(0,1,0,1),13.25),"Rank-four filename PONNI tensor load is incorrect");
  std::remove(path.c_str());
}

void test_ponni_file_rejections(Check & check) {
  std::string const valid_path = "ponni_rejection_valid.ponni";
  std::vector<double> const values{1.0,2.0,3.0,4.0};
  std::vector<ponni::PonniTensorSpec> const specs{
      {"first","F64",{2},0},
      {"second","F64",{2},2},
  };
  std::string error;
  check(ponni::write_ponni_file(valid_path,specs,"rejection-model",values.data(),&error,"test"),
        "Could not create the PONNI rejection-test fixture: " + error);

  auto reject_bytes = [&](std::string const & label, std::vector<unsigned char> const & bytes,
                          std::string const & expected_error) {
    std::string const path = "ponni_rejection_" + label + ".ponni";
    write_bytes(path,bytes);
    ponni::PonniFile file;
    std::string load_error;
    bool const rejected = !file.load(path,&load_error);
    check(rejected,label + " PONNI file was unexpectedly accepted");
    check(expected_error.empty() || load_error.find(expected_error) != std::string::npos,
          label + " PONNI rejection did not identify " + expected_error + ": " + load_error);
    std::remove(path.c_str());
  };

  auto const valid_bytes = read_bytes(valid_path);
  reject_bytes("short",std::vector<unsigned char>{0,1,2},"shorter");

  auto truncated_header = valid_bytes;
  truncated_header.resize(10);
  reject_bytes("truncated_header",truncated_header,"header size");

  auto invalid_json = valid_bytes;
  invalid_json[8] = '[';
  reject_bytes("invalid_json",invalid_json,"JSON");
  reject_bytes("nonobject_root",make_safetensors_bytes("[]"),"root must be an object");
  reject_bytes("duplicate_key",make_safetensors_bytes("{\"x\":{},\"x\":{}}"),"duplicate");
  reject_bytes("missing_metadata",make_safetensors_bytes("{}"),"no Safetensors metadata");
  reject_bytes("incomplete_descriptor",make_safetensors_bytes("{\"x\":{}}"),"incomplete");

  std::string const placeholder_metadata =
      "\"__metadata__\":{"
      "\"ponni.profile_version\":\"1\"," 
      "\"ponni.model_fingerprint\":\"x\"," 
      "\"ponni.schema_fingerprint\":\"x\"," 
      "\"ponni.payload_checksum_fnv1a64\":\"x\"}";
  reject_bytes("invalid_shape",
               make_safetensors_bytes("{" + placeholder_metadata +
                                      ",\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}",4),
               "invalid Safetensors shape");
  reject_bytes("byte_length",
               make_safetensors_bytes("{" + placeholder_metadata +
                                      ",\"x\":{\"dtype\":\"F64\",\"shape\":[2],\"data_offsets\":[0,8]}}",8),
               "dtype, shape, and byte length");
  reject_bytes("payload_overlap",
               make_safetensors_bytes(
                   "{" + placeholder_metadata +
                   ",\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                   "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",4),
               "hole or overlap");

  auto nonstring_metadata = valid_bytes;
  check(replace_bytes(nonstring_metadata,"\"ponni.profile_version\":\"1\"",
                      "\"ponni.profile_version\": 1 "),
        "Could not construct the non-string metadata fixture");
  reject_bytes("nonstring_metadata",nonstring_metadata,"metadata values");

  auto unsupported_version = valid_bytes;
  check(replace_bytes(unsupported_version,"\"ponni.profile_version\":\"1\"",
                      "\"ponni.profile_version\":\"2\""),
        "Could not construct the unsupported-version fixture");
  reject_bytes("unsupported_version",unsupported_version,"profile version");

  auto unknown_dtype = valid_bytes;
  check(replace_bytes(unknown_dtype,"\"dtype\":\"F64\"","\"dtype\":\"BAD\""),
        "Could not construct the unknown-dtype fixture");
  reject_bytes("unknown_dtype",unknown_dtype,"dtype, shape, and byte length");

  reject_bytes("payload_gap",
               make_safetensors_bytes(
                   "{" + placeholder_metadata +
                   ",\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                   "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[5,9]}}",9),
               "hole or overlap");

  auto truncated_payload = valid_bytes;
  truncated_payload.pop_back();
  reject_bytes("truncated_payload",truncated_payload,"dtype, shape, and byte length");

  auto trailing_payload = valid_bytes;
  trailing_payload.push_back(0);
  reject_bytes("trailing_payload",trailing_payload,"trailing bytes");

  auto corrupt_payload = valid_bytes;
  corrupt_payload.back() ^= 1;
  reject_bytes("checksum",corrupt_payload,"checksum");

  ponni::PonniFile valid;
  error.clear();
  check(valid.load(valid_path,&error),"Valid PONNI rejection fixture did not load: " + error);
  auto expect_validation_failure = [&](std::vector<ponni::PonniTensorSpec> const & expected,
                                       std::string const & fingerprint, std::string const & message) {
    std::string validation_error;
    check(!valid.validate(expected,fingerprint,&validation_error),message + " was unexpectedly accepted");
  };
  expect_validation_failure({specs[0]},"rejection-model","Wrong tensor count");
  expect_validation_failure({{"missing","F64",{2},0},specs[1]},"rejection-model","Missing tensor name");
  expect_validation_failure({{"first","F32",{2},0},specs[1]},"rejection-model","Wrong tensor dtype");
  expect_validation_failure({{"first","F64",{1,2},0},specs[1]},"rejection-model","Wrong tensor shape");
  expect_validation_failure(specs,"foreign-model","Wrong model fingerprint");

  bool missing_tensor_rejected = false;
  bool wrong_rank_rejected = false;
  try {
    auto ignored = ponni::load_ponni_tensor<1,Kokkos::HostSpace>(valid,"missing");
    (void) ignored;
  } catch (std::runtime_error const &) {
    missing_tensor_rejected = true;
  }
  try {
    auto ignored = ponni::load_ponni_tensor<2,Kokkos::HostSpace>(valid,"first");
    (void) ignored;
  } catch (std::runtime_error const &) {
    wrong_rank_rejected = true;
  }
  check(missing_tensor_rejected,"load_ponni_tensor accepted a missing tensor");
  check(wrong_rank_rejected,"load_ponni_tensor accepted an incorrect rank");
  std::remove(valid_path.c_str());
}

void test_orthogonal_initializer(Check & check) {
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
  auto check_gram = [&](int rows, int columns, bool columns_are_vectors, std::string const & label) {
    Kokkos::View<float**,Kokkos::LayoutRight,MemorySpace> matrix(label,rows,columns);
    Kokkos::View<float**,Kokkos::LayoutRight,MemorySpace> repeated(label + "_repeated",rows,columns);
    ponni::Initializer_Orthogonal<float> const initializer(1.5f,8128 + rows);
    initializer.fill(matrix);
    initializer.fill(repeated);
    auto host = ponni::create_host_copy(matrix);
    auto repeated_host = ponni::create_host_copy(repeated);
    bool deterministic = true;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) deterministic = deterministic && host(i,j) == repeated_host(i,j);
    }
    check(deterministic,label + " is not deterministic for a nonzero seed");
    int const vectors = columns_are_vectors ? columns : rows;
    int const elements = columns_are_vectors ? rows : columns;
    for (int left = 0; left < vectors; left++) {
      for (int right = 0; right < vectors; right++) {
        double dot = 0;
        for (int element = 0; element < elements; element++) {
          double const a = columns_are_vectors ? host(element,left) : host(left,element);
          double const b = columns_are_vectors ? host(element,right) : host(right,element);
          dot += a * b;
        }
        double const expected = left == right ? 2.25 : 0.0;
        check(std::abs(dot - expected) < 2.e-4,label + " orthogonality or gain is incorrect");
      }
    }
  };
  check_gram(6,3,true,"tall orthogonal initializer");
  check_gram(3,6,false,"wide orthogonal initializer");

  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> vector("orthogonal_rank1",16);
  Kokkos::View<float*,Kokkos::LayoutRight,MemorySpace> repeated_vector("orthogonal_rank1_repeated",16);
  ponni::Initializer_Orthogonal<float> const initializer(1.0f,991);
  initializer.fill(vector);
  initializer.fill(repeated_vector);
  auto host = ponni::create_host_copy(vector);
  auto repeated_host = ponni::create_host_copy(repeated_vector);
  bool finite = true;
  bool deterministic = true;
  double magnitude = 0;
  for (int i = 0; i < 16; i++) {
    finite = finite && std::isfinite(host(i));
    deterministic = deterministic && host(i) == repeated_host(i);
    magnitude += std::abs(host(i));
  }
  check(finite && magnitude > 0,"Orthogonal initializer rank-one fallback is incorrect");
  check(deterministic,"Orthogonal initializer rank-one fallback is not deterministic for a nonzero seed");
}

void test_custom_layer_contract(Check & check) {
  using Barrier = TestBarrier<float,2>;
  static_assert(ponni::LayerTraits<Barrier>::fusion_kind == ponni::LayerFusionKind::barrier);
  static_assert(ponni::is_dense_layer_v<TestDense<float,2>>);
  static_assert(ponni::is_pointwise_layer_v<TestIndexedPointwise<float,2>>);

  auto model = ponni::create_inference_model(
      TestDense<float,2>(), TestIndexedPointwise<float,2>(), TestBarrier<float,2>(), TestDense<float,2>());
  using Model = decltype(model);
  static_assert(Model::get_num_fused_dense_blocks() == 2);
  typename Model::real2d input("custom_layer_input",2,1);
  auto input_host = Kokkos::create_mirror_view(input);
  input_host(0,0) = 1.0f;
  input_host(1,0) = 2.0f;
  Kokkos::deep_copy(input,input_host);
  auto output_host = ponni::create_host_copy(model.forward_batch_parallel(input));
  check(close(output_host(0,0),4.5) && close(output_host(1,0),-4.5),
        "Custom dense/pointwise/barrier View model produced an incorrect answer");

  Kokkos::View<float*,Kokkos::LayoutRight,typename Kokkos::DefaultExecutionSpace::memory_space>
      result("custom_sarray_result",2);
  auto const params = model.params;
  Kokkos::parallel_for("Custom layer SArray model",1,KOKKOS_LAMBDA(int) {
    ponni::SArray<float,2> local_input;
    ponni::SArray<float,2> local_output;
    local_input(0) = 1.0f;
    local_input(1) = 2.0f;
    Model::forward_batch_parallel_in_kernel(local_input,local_output,params);
    result(0) = local_output(0);
    result(1) = local_output(1);
  });
  auto result_host = ponni::create_host_copy(result);
  check(close(result_host(0),4.5) && close(result_host(1),-4.5),
        "Custom dense/pointwise/barrier SArray model produced an incorrect answer");

  // Supplying only an execution-space instance exercises memory inference and
  // forces every custom test layer through its copy_to_memory_space contract.
  using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  auto host_model = ponni::create_inference_model(
      HostExecutionSpace(),
      TestDense<float,2>(), TestIndexedPointwise<float,2>(), TestBarrier<float,2>(), TestDense<float,2>());
  static_assert(std::is_same_v<typename decltype(host_model)::memory_space,typename HostExecutionSpace::memory_space>);
  Kokkos::View<float**,Kokkos::LayoutRight,typename HostExecutionSpace::memory_space> host_input("custom_host_input",2,1);
  host_input(0,0) = 1.0f;
  host_input(1,0) = 2.0f;
  auto host_output = host_model.forward_batch_parallel(host_input);
  check(close(host_output(0,0),4.5) && close(host_output(1,0),-4.5),
        "Execution-space-only custom-layer model produced an incorrect answer");
}

} // namespace

int main(int argc, char ** argv) {
  Kokkos::initialize(argc,argv);
  Check check;
  {
    test_sarray_api(check);
    test_builtin_sarray_and_single_layer(check);
    test_saved_state_model_edges(check);
    test_precision_factories(check);
    test_ponni_tensor_ranks(check);
    test_ponni_file_rejections(check);
    test_orthogonal_initializer(check);
    test_custom_layer_contract(check);
  }
  Kokkos::finalize();
  if (!check.passed) return 1;
  std::cout << "template_coverage passed" << std::endl;
  return 0;
}
