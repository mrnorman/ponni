#pragma once

// A deliberately small, dependency-free JSON parser for Safetensors headers.
// It implements the complete JSON value grammar, rejects duplicate object
// keys, reports byte offsets, limits nesting, and decodes Unicode escapes. It
// is host-only; no standard-library state from this file enters device code.

#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ponni::detail {

  struct JsonValue {
    enum class Kind { null_value, boolean, number, string, array, object };

    Kind kind = Kind::null_value;
    bool boolean = false;
    std::string text;
    std::vector<JsonValue> array;
    std::map<std::string,JsonValue> object;

    JsonValue const * find(std::string const & key) const {
      auto const iterator = object.find(key);
      return iterator == object.end() ? nullptr : &iterator->second;
    }
  };

  class JsonParser {
    char const * begin_;
    char const * current_;
    char const * end_;
    std::string error_;
    int depth_ = 0;
    int static constexpr max_depth = 64;

    void fail(std::string const & message) {
      if (error_.empty()) {
        error_ = message + " at JSON byte " + std::to_string(current_ - begin_);
      }
    }

    void whitespace() {
      while (current_ != end_ && (*current_ == ' ' || *current_ == '\t' ||
                                  *current_ == '\n' || *current_ == '\r')) current_++;
    }

    bool consume(char expected) {
      if (current_ == end_ || *current_ != expected) {
        fail(std::string("expected '") + expected + "'");
        return false;
      }
      current_++;
      return true;
    }

    static int hex_digit(char value) {
      if (value >= '0' && value <= '9') return value - '0';
      if (value >= 'a' && value <= 'f') return value - 'a' + 10;
      if (value >= 'A' && value <= 'F') return value - 'A' + 10;
      return -1;
    }

    bool unicode_escape(std::uint32_t & codepoint) {
      if (end_ - current_ < 4) {
        fail("incomplete Unicode escape");
        return false;
      }
      codepoint = 0;
      for (int i = 0; i < 4; i++) {
        int const digit = hex_digit(*current_++);
        if (digit < 0) {
          fail("invalid Unicode escape");
          return false;
        }
        codepoint = codepoint * 16 + static_cast<std::uint32_t>(digit);
      }
      return true;
    }

    static void append_utf8(std::string & output, std::uint32_t codepoint) {
      if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
      } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      }
    }

    bool string(std::string & output) {
      if (!consume('"')) return false;
      while (current_ != end_) {
        unsigned char const value = static_cast<unsigned char>(*current_++);
        if (value == '"') return true;
        if (value < 0x20) {
          fail("unescaped control character in string");
          return false;
        }
        if (value != '\\') {
          output.push_back(static_cast<char>(value));
          continue;
        }
        if (current_ == end_) {
          fail("incomplete string escape");
          return false;
        }
        char const escaped = *current_++;
        if      (escaped == '"') output.push_back('"');
        else if (escaped == '\\') output.push_back('\\');
        else if (escaped == '/')  output.push_back('/');
        else if (escaped == 'b')  output.push_back('\b');
        else if (escaped == 'f')  output.push_back('\f');
        else if (escaped == 'n')  output.push_back('\n');
        else if (escaped == 'r')  output.push_back('\r');
        else if (escaped == 't')  output.push_back('\t');
        else if (escaped == 'u') {
          std::uint32_t codepoint = 0;
          if (!unicode_escape(codepoint)) return false;
          if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            if (end_ - current_ < 6 || current_[0] != '\\' || current_[1] != 'u') {
              fail("missing low Unicode surrogate");
              return false;
            }
            current_ += 2;
            std::uint32_t low = 0;
            if (!unicode_escape(low)) return false;
            if (low < 0xdc00 || low > 0xdfff) {
              fail("invalid low Unicode surrogate");
              return false;
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
          } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
            fail("unpaired low Unicode surrogate");
            return false;
          }
          append_utf8(output,codepoint);
        } else {
          fail("invalid string escape");
          return false;
        }
      }
      fail("unterminated string");
      return false;
    }

    bool literal(char const * expected) {
      for (char const * pointer = expected; *pointer != '\0'; pointer++) {
        if (current_ == end_ || *current_++ != *pointer) {
          fail(std::string("expected ") + expected);
          return false;
        }
      }
      return true;
    }

    bool number(JsonValue & value) {
      char const * start = current_;
      if (current_ != end_ && *current_ == '-') current_++;
      if (current_ == end_) {
        fail("incomplete number");
        return false;
      }
      if (*current_ == '0') {
        current_++;
      } else if (*current_ >= '1' && *current_ <= '9') {
        while (current_ != end_ && std::isdigit(static_cast<unsigned char>(*current_))) current_++;
      } else {
        fail("invalid number");
        return false;
      }
      if (current_ != end_ && *current_ == '.') {
        current_++;
        if (current_ == end_ || !std::isdigit(static_cast<unsigned char>(*current_))) {
          fail("invalid fractional number");
          return false;
        }
        while (current_ != end_ && std::isdigit(static_cast<unsigned char>(*current_))) current_++;
      }
      if (current_ != end_ && (*current_ == 'e' || *current_ == 'E')) {
        current_++;
        if (current_ != end_ && (*current_ == '+' || *current_ == '-')) current_++;
        if (current_ == end_ || !std::isdigit(static_cast<unsigned char>(*current_))) {
          fail("invalid exponent");
          return false;
        }
        while (current_ != end_ && std::isdigit(static_cast<unsigned char>(*current_))) current_++;
      }
      value.kind = JsonValue::Kind::number;
      value.text.assign(start,current_);
      return true;
    }

    bool array(JsonValue & value) {
      if (!consume('[')) return false;
      value.kind = JsonValue::Kind::array;
      whitespace();
      if (current_ != end_ && *current_ == ']') {
        current_++;
        return true;
      }
      while (current_ != end_) {
        JsonValue child;
        if (!parse_value(child)) return false;
        value.array.push_back(std::move(child));
        whitespace();
        if (current_ != end_ && *current_ == ']') {
          current_++;
          return true;
        }
        if (!consume(',')) return false;
        whitespace();
      }
      fail("unterminated array");
      return false;
    }

    bool object(JsonValue & value) {
      if (!consume('{')) return false;
      value.kind = JsonValue::Kind::object;
      whitespace();
      if (current_ != end_ && *current_ == '}') {
        current_++;
        return true;
      }
      while (current_ != end_) {
        std::string key;
        if (!string(key)) return false;
        whitespace();
        if (!consume(':')) return false;
        whitespace();
        JsonValue child;
        if (!parse_value(child)) return false;
        if (!value.object.emplace(key,std::move(child)).second) {
          fail("duplicate object key " + key);
          return false;
        }
        whitespace();
        if (current_ != end_ && *current_ == '}') {
          current_++;
          return true;
        }
        if (!consume(',')) return false;
        whitespace();
      }
      fail("unterminated object");
      return false;
    }

    bool parse_value(JsonValue & value) {
      if (++depth_ > max_depth) {
        fail("JSON nesting limit exceeded");
        return false;
      }
      whitespace();
      bool result = false;
      if (current_ == end_) {
        fail("expected JSON value");
      } else if (*current_ == '{') {
        result = object(value);
      } else if (*current_ == '[') {
        result = array(value);
      } else if (*current_ == '"') {
        value.kind = JsonValue::Kind::string;
        result = string(value.text);
      } else if (*current_ == 't') {
        value.kind = JsonValue::Kind::boolean;
        value.boolean = true;
        result = literal("true");
      } else if (*current_ == 'f') {
        value.kind = JsonValue::Kind::boolean;
        value.boolean = false;
        result = literal("false");
      } else if (*current_ == 'n') {
        value.kind = JsonValue::Kind::null_value;
        result = literal("null");
      } else {
        result = number(value);
      }
      depth_--;
      return result;
    }

  public:
    JsonParser(char const * data, std::size_t size) : begin_(data), current_(data), end_(data + size) {}

    bool parse(JsonValue & value, std::string & error) {
      bool const result = parse_value(value);
      whitespace();
      if (result && current_ != end_) fail("unexpected characters after JSON value");
      error = error_;
      return result && error_.empty();
    }
  };

} // namespace ponni::detail
