#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ponni {
namespace test {

// Reference files are produced by the Python exporters. Values are flattened
// in feature-major order so the C++ tests can copy directly into LayoutRight
// views with (feature,batch) indexing.
struct ReferenceCase {
  int batch_size;
  std::vector<float> inputs;
  std::vector<float> outputs;
};

struct ReferenceData {
  int num_inputs;
  int num_outputs;
  std::vector<ReferenceCase> cases;
};

inline ReferenceData load_reference(std::string const & path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open reference file: " + path);

  int num_cases;
  ReferenceData data;
  stream >> num_cases >> data.num_inputs >> data.num_outputs;
  for (int icase = 0; icase < num_cases; icase++) {
    ReferenceCase reference;
    stream >> reference.batch_size;
    reference.inputs.resize(data.num_inputs * reference.batch_size);
    reference.outputs.resize(data.num_outputs * reference.batch_size);
    for (float & value : reference.inputs) stream >> value;
    for (float & value : reference.outputs) stream >> value;
    data.cases.push_back(std::move(reference));
  }
  if (!stream) throw std::runtime_error("malformed reference file: " + path);
  return data;
}

}  // namespace test
}  // namespace ponni
