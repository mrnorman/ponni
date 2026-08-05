#pragma once

// Host-side reader/writer for the PONNI profile of Safetensors. Standard
// Safetensors tools can read the files; PONNI additionally requires exact model
// and tensor-schema fingerprints plus an FNV-1a checksum over the data buffer.

#include "ponni_json.h"
#include "ponni_kokkos_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ponni {

  inline std::uint64_t ponni_fnv1a64(unsigned char const * data, std::size_t size) {
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (std::size_t i = 0; i < size; i++) {
      value ^= data[i];
      value *= UINT64_C(1099511628211);
    }
    return value;
  }

  inline std::string ponni_fnv1a64_string(std::uint64_t value) {
    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
  }

  struct PonniTensorDescriptor {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t data_begin = 0;
    std::size_t data_end = 0;
  };

  // A generated model uses the source offset to scatter tensors from the
  // name-sorted Safetensors payload into its compile-time flattened storage.
  struct PonniTensorSpec {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t source_element_offset = 0;
  };

  namespace detail {

    inline bool parse_size(JsonValue const & value, std::size_t & result) {
      if (value.kind != JsonValue::Kind::number || value.text.empty()) return false;
      std::size_t parsed = 0;
      for (char character : value.text) {
        if (character < '0' || character > '9') return false;
        std::size_t const digit = static_cast<std::size_t>(character - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10) return false;
        parsed = parsed * 10 + digit;
      }
      result = parsed;
      return true;
    }

    inline std::size_t dtype_bytes(std::string const & dtype) {
      if (dtype == "BOOL" || dtype == "I8" || dtype == "U8") return 1;
      if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
      if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
      if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
      return 0;
    }

    inline bool checked_elements(std::vector<std::size_t> const & shape, std::size_t & elements) {
      elements = 1;
      for (std::size_t dimension : shape) {
        if (dimension != 0 && elements > std::numeric_limits<std::size_t>::max() / dimension) return false;
        elements *= dimension;
      }
      return true;
    }

    inline std::uint64_t little_u64(unsigned char const * bytes) {
      std::uint64_t value = 0;
      for (int i = 0; i < 8; i++) value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
      return value;
    }

    inline void append_little_u64(std::vector<unsigned char> & bytes, std::uint64_t value) {
      for (int i = 0; i < 8; i++) bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }

    inline std::string json_escape(std::string const & value) {
      std::ostringstream stream;
      stream << '"';
      for (unsigned char character : value) {
        if      (character == '"')  stream << "\\\"";
        else if (character == '\\') stream << "\\\\";
        else if (character == '\b') stream << "\\b";
        else if (character == '\f') stream << "\\f";
        else if (character == '\n') stream << "\\n";
        else if (character == '\r') stream << "\\r";
        else if (character == '\t') stream << "\\t";
        else if (character < 0x20) {
          stream << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(character);
        } else {
          stream << static_cast<char>(character);
        }
      }
      stream << '"';
      return stream.str();
    }

    inline std::string schema_fingerprint(std::vector<PonniTensorDescriptor> descriptors) {
      std::sort(descriptors.begin(),descriptors.end(),
                [](auto const & left, auto const & right) { return left.name < right.name; });
      std::ostringstream canonical;
      canonical << "ponni-tensor-schema-v1\n";
      for (auto const & descriptor : descriptors) {
        canonical << descriptor.name << '\t' << descriptor.dtype << '\t';
        for (std::size_t i = 0; i < descriptor.shape.size(); i++) {
          if (i != 0) canonical << ',';
          canonical << descriptor.shape[i];
        }
        canonical << '\n';
      }
      std::string const text = canonical.str();
      auto const * bytes = reinterpret_cast<unsigned char const *>(text.data());
      return ponni_fnv1a64_string(ponni_fnv1a64(bytes,text.size()));
    }

    template <class Scalar>
    inline std::string safetensors_dtype() {
      if constexpr (std::is_same_v<Scalar,float>) return "F32";
      else if constexpr (std::is_same_v<Scalar,double>) return "F64";
      else return "";
    }

    template <class Scalar>
    inline void append_scalar(std::vector<unsigned char> & payload, Scalar value) {
      static_assert(std::is_same_v<Scalar,float> || std::is_same_v<Scalar,double>);
      unsigned char bytes[sizeof(Scalar)];
      std::memcpy(bytes,&value,sizeof(Scalar));
      std::uint16_t const endian_probe = 1;
      if (*reinterpret_cast<unsigned char const *>(&endian_probe) == 1) {
        payload.insert(payload.end(),bytes,bytes + sizeof(Scalar));
      } else {
        for (std::size_t i = 0; i < sizeof(Scalar); i++) payload.push_back(bytes[sizeof(Scalar) - 1 - i]);
      }
    }

    template <class Scalar>
    inline Scalar read_scalar(unsigned char const * bytes) {
      static_assert(std::is_same_v<Scalar,float> || std::is_same_v<Scalar,double>);
      unsigned char native[sizeof(Scalar)];
      std::uint16_t const endian_probe = 1;
      if (*reinterpret_cast<unsigned char const *>(&endian_probe) == 1) {
        std::memcpy(native,bytes,sizeof(Scalar));
      } else {
        for (std::size_t i = 0; i < sizeof(Scalar); i++) native[i] = bytes[sizeof(Scalar) - 1 - i];
      }
      Scalar value;
      std::memcpy(&value,native,sizeof(Scalar));
      return value;
    }

  } // namespace detail

  class PonniFile {
    std::vector<unsigned char> bytes_;
    std::size_t data_offset_ = 0;
    std::map<std::string,std::string> metadata_;
    std::map<std::string,PonniTensorDescriptor> tensors_;

    bool fail(std::string const & message, std::string * error) {
      if (error != nullptr) *error = message;
      return false;
    }

  public:
    bool load(std::string const & path, std::string * error = nullptr) {
      bytes_.clear();
      metadata_.clear();
      tensors_.clear();
      data_offset_ = 0;

      std::ifstream stream(path,std::ios::binary | std::ios::ate);
      if (!stream) return fail("cannot open PONNI file: " + path,error);
      std::streamsize const file_size = stream.tellg();
      if (file_size < 0) return fail("cannot determine PONNI file size: " + path,error);
      stream.seekg(0,std::ios::beg);
      bytes_.resize(static_cast<std::size_t>(file_size));
      if (!stream.read(reinterpret_cast<char *>(bytes_.data()),file_size)) {
        return fail("cannot read PONNI file: " + path,error);
      }
      if (bytes_.size() < 10) return fail("PONNI file is shorter than a Safetensors header",error);
      std::uint64_t const header_size_u64 = detail::little_u64(bytes_.data());
      std::size_t constexpr max_header_size = 100 * 1024 * 1024;
      if (header_size_u64 < 2 || header_size_u64 > max_header_size ||
          header_size_u64 > bytes_.size() - 8) {
        return fail("PONNI file has an invalid Safetensors header size",error);
      }
      std::size_t const header_size = static_cast<std::size_t>(header_size_u64);
      data_offset_ = 8 + header_size;

      detail::JsonValue root;
      detail::JsonParser parser(reinterpret_cast<char const *>(bytes_.data() + 8),header_size);
      std::string parse_error;
      if (!parser.parse(root,parse_error)) return fail("invalid Safetensors JSON: " + parse_error,error);
      if (root.kind != detail::JsonValue::Kind::object) {
        return fail("Safetensors header root must be an object",error);
      }

      for (auto const & [name, value] : root.object) {
        if (name == "__metadata__") {
          if (value.kind != detail::JsonValue::Kind::object) {
            return fail("Safetensors metadata must be an object",error);
          }
          for (auto const & [key, item] : value.object) {
            if (item.kind != detail::JsonValue::Kind::string) {
              return fail("Safetensors metadata values must be strings",error);
            }
            metadata_.emplace(key,item.text);
          }
          continue;
        }
        if (value.kind != detail::JsonValue::Kind::object) {
          return fail("Safetensors tensor descriptor for " + name + " must be an object",error);
        }
        auto const * dtype = value.find("dtype");
        auto const * shape = value.find("shape");
        auto const * offsets = value.find("data_offsets");
        if (dtype == nullptr || shape == nullptr || offsets == nullptr ||
            dtype->kind != detail::JsonValue::Kind::string ||
            shape->kind != detail::JsonValue::Kind::array ||
            offsets->kind != detail::JsonValue::Kind::array || offsets->array.size() != 2) {
          return fail("incomplete Safetensors descriptor for " + name,error);
        }
        PonniTensorDescriptor descriptor;
        descriptor.name = name;
        descriptor.dtype = dtype->text;
        for (auto const & dimension : shape->array) {
          std::size_t parsed = 0;
          if (!detail::parse_size(dimension,parsed)) {
            return fail("invalid Safetensors shape for " + name,error);
          }
          descriptor.shape.push_back(parsed);
        }
        if (!detail::parse_size(offsets->array[0],descriptor.data_begin) ||
            !detail::parse_size(offsets->array[1],descriptor.data_end)) {
          return fail("invalid Safetensors offsets for " + name,error);
        }
        std::size_t const scalar_bytes = detail::dtype_bytes(descriptor.dtype);
        std::size_t elements = 0;
        if (scalar_bytes == 0 || !detail::checked_elements(descriptor.shape,elements) ||
            elements > std::numeric_limits<std::size_t>::max() / scalar_bytes ||
            descriptor.data_begin > descriptor.data_end ||
            descriptor.data_end > bytes_.size() - data_offset_ ||
            descriptor.data_end - descriptor.data_begin != elements * scalar_bytes) {
          return fail("Safetensors dtype, shape, and byte length disagree for " + name,error);
        }
        tensors_.emplace(name,std::move(descriptor));
      }

      if (metadata_.empty()) return fail("PONNI file has no Safetensors metadata",error);
      auto require_metadata = [&](std::string const & key) -> std::string const * {
        auto const iterator = metadata_.find(key);
        return iterator == metadata_.end() ? nullptr : &iterator->second;
      };
      auto const * version = require_metadata("ponni.profile_version");
      auto const * model = require_metadata("ponni.model_fingerprint");
      auto const * schema = require_metadata("ponni.schema_fingerprint");
      auto const * checksum = require_metadata("ponni.payload_checksum_fnv1a64");
      if (version == nullptr || model == nullptr || schema == nullptr || checksum == nullptr) {
        return fail("PONNI Safetensors metadata is incomplete",error);
      }
      if (*version != "1") return fail("unsupported PONNI profile version " + *version,error);

      std::size_t cursor = 0;
      std::vector<PonniTensorDescriptor> physical;
      for (auto const & [_, descriptor] : tensors_) physical.push_back(descriptor);
      std::sort(physical.begin(),physical.end(),[](auto const & left, auto const & right) {
        if (left.data_begin != right.data_begin) return left.data_begin < right.data_begin;
        return left.data_end < right.data_end;
      });
      for (auto const & descriptor : physical) {
        if (descriptor.data_begin != cursor) {
          return fail("Safetensors payload has a hole or overlap before " + descriptor.name,error);
        }
        cursor = descriptor.data_end;
      }
      if (cursor != bytes_.size() - data_offset_) {
        return fail("Safetensors payload contains unindexed trailing bytes",error);
      }
      if (*schema != detail::schema_fingerprint(physical)) {
        return fail("PONNI tensor-schema fingerprint does not match the Safetensors descriptors",error);
      }
      unsigned char const * payload = bytes_.data() + data_offset_;
      std::string const actual_checksum = ponni_fnv1a64_string(
          ponni_fnv1a64(payload,bytes_.size() - data_offset_));
      if (*checksum != actual_checksum) return fail("PONNI payload checksum mismatch",error);
      return true;
    }

    PonniTensorDescriptor const * find(std::string const & name) const {
      auto const iterator = tensors_.find(name);
      return iterator == tensors_.end() ? nullptr : &iterator->second;
    }

    std::string const * metadata(std::string const & key) const {
      auto const iterator = metadata_.find(key);
      return iterator == metadata_.end() ? nullptr : &iterator->second;
    }

    unsigned char const * tensor_data(PonniTensorDescriptor const & tensor) const {
      return bytes_.data() + data_offset_ + tensor.data_begin;
    }

    std::size_t tensor_count() const { return tensors_.size(); }

    bool validate(std::vector<PonniTensorSpec> const & expected, std::string const & model_fingerprint,
                  std::string * error = nullptr) const {
      auto fail_validation = [&](std::string const & message) {
        if (error != nullptr) *error = message;
        return false;
      };
      auto const * file_model = metadata("ponni.model_fingerprint");
      if (file_model == nullptr || *file_model != model_fingerprint) {
        return fail_validation("PONNI model fingerprint does not match this model");
      }
      if (expected.size() != tensors_.size()) {
        return fail_validation("PONNI tensor count does not match this model");
      }
      for (auto const & spec : expected) {
        auto const * tensor = find(spec.name);
        if (tensor == nullptr) return fail_validation("PONNI file is missing tensor " + spec.name);
        if (tensor->dtype != spec.dtype) return fail_validation("PONNI tensor dtype mismatch for " + spec.name);
        if (tensor->shape != spec.shape) return fail_validation("PONNI tensor shape mismatch for " + spec.name);
      }
      return true;
    }
  };

  template <class Scalar>
  inline bool write_ponni_file(std::string const & path, std::vector<PonniTensorSpec> specs,
                               std::string const & model_fingerprint, Scalar const * source,
                               std::string * error = nullptr, std::string const & target = "ponni") {
    static_assert(std::is_same_v<Scalar,float> || std::is_same_v<Scalar,double>);
    auto fail = [&](std::string const & message) {
      if (error != nullptr) *error = message;
      return false;
    };
    std::sort(specs.begin(),specs.end(),[](auto const & left, auto const & right) { return left.name < right.name; });
    std::string const dtype = detail::safetensors_dtype<Scalar>();
    std::vector<unsigned char> payload;
    std::vector<PonniTensorDescriptor> descriptors;
    for (auto const & spec : specs) {
      if (spec.dtype != dtype) return fail("PONNI tensor dtype does not match the supplied scalar type");
      std::size_t elements = 0;
      if (!detail::checked_elements(spec.shape,elements)) return fail("PONNI tensor shape overflows size_t");
      PonniTensorDescriptor descriptor{spec.name,spec.dtype,spec.shape,payload.size(),0};
      for (std::size_t i = 0; i < elements; i++) {
        detail::append_scalar(payload,source[spec.source_element_offset + i]);
      }
      descriptor.data_end = payload.size();
      descriptors.push_back(std::move(descriptor));
    }

    std::string const checksum = ponni_fnv1a64_string(ponni_fnv1a64(payload.data(),payload.size()));
    std::string const schema = detail::schema_fingerprint(descriptors);
    std::ostringstream header;
    header << "{\"__metadata__\":{"
           << "\"ponni.profile_version\":\"1\"," 
           << "\"ponni.model_fingerprint\":" << detail::json_escape(model_fingerprint) << ','
           << "\"ponni.schema_fingerprint\":" << detail::json_escape(schema) << ','
           << "\"ponni.payload_checksum_fnv1a64\":" << detail::json_escape(checksum) << ','
           << "\"ponni.source_framework\":\"ponni-cpp\"," 
           << "\"ponni.target\":" << detail::json_escape(target) << '}';
    for (auto const & descriptor : descriptors) {
      header << ',' << detail::json_escape(descriptor.name) << ":{\"dtype\":"
             << detail::json_escape(descriptor.dtype) << ",\"shape\":[";
      for (std::size_t i = 0; i < descriptor.shape.size(); i++) {
        if (i != 0) header << ',';
        header << descriptor.shape[i];
      }
      header << "],\"data_offsets\":[" << descriptor.data_begin << ',' << descriptor.data_end << "]}";
    }
    header << '}';
    std::string header_text = header.str();
    header_text.append((8 - header_text.size() % 8) % 8,' ');
    std::vector<unsigned char> bytes;
    detail::append_little_u64(bytes,header_text.size());
    bytes.insert(bytes.end(),header_text.begin(),header_text.end());
    bytes.insert(bytes.end(),payload.begin(),payload.end());

    std::ofstream stream(path,std::ios::binary | std::ios::trunc);
    if (!stream) return fail("cannot open PONNI output file: " + path);
    stream.write(reinterpret_cast<char const *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (!stream) return fail("cannot write PONNI output file: " + path);
    return true;
  }

  template <int N, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  inline Kokkos::View<typename TypeIntToViewType<float,N>::type,Kokkos::LayoutRight,MemorySpace>
  load_ponni_tensor(PonniFile const & file, std::string const & tensor_name) {
    static_assert(N >= 1 && N <= 4,"load_ponni_tensor supports ranks one through four");
    using value_type = typename TypeIntToViewType<float,N>::type;
    using HostView = Kokkos::View<value_type,Kokkos::LayoutRight,Kokkos::HostSpace>;
    auto const * tensor = file.find(tensor_name);
    if (tensor == nullptr) throw std::runtime_error("PONNI file is missing tensor " + tensor_name);
    if (tensor->shape.size() != N) throw std::runtime_error("PONNI tensor rank mismatch for " + tensor_name);
    if (tensor->dtype != "F32" && tensor->dtype != "F64") {
      throw std::runtime_error("PONNI tensor must use F32 or F64: " + tensor_name);
    }
    HostView host;
    if constexpr (N == 1) host = HostView(tensor_name,tensor->shape[0]);
    if constexpr (N == 2) host = HostView(tensor_name,tensor->shape[0],tensor->shape[1]);
    if constexpr (N == 3) host = HostView(tensor_name,tensor->shape[0],tensor->shape[1],tensor->shape[2]);
    if constexpr (N == 4) host = HostView(tensor_name,tensor->shape[0],tensor->shape[1],tensor->shape[2],tensor->shape[3]);
    std::size_t elements = 0;
    detail::checked_elements(tensor->shape,elements);
    unsigned char const * data = file.tensor_data(*tensor);
    if (tensor->dtype == "F32") {
      for (std::size_t i = 0; i < elements; i++) host.data()[i] = detail::read_scalar<float>(data + 4 * i);
    } else {
      for (std::size_t i = 0; i < elements; i++) host.data()[i] = static_cast<float>(detail::read_scalar<double>(data + 8 * i));
    }
    return create_memory_space_copy(host,MemorySpace());
  }

  template <int N, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
  inline Kokkos::View<typename TypeIntToViewType<float,N>::type,Kokkos::LayoutRight,MemorySpace>
  load_ponni_tensor(std::string const & path, std::string const & tensor_name) {
    PonniFile file;
    std::string error;
    if (!file.load(path,&error)) throw std::runtime_error(error);
    return load_ponni_tensor<N,MemorySpace>(file,tensor_name);
  }

} // namespace ponni
