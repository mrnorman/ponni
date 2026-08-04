#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace ponni {

// Predicates for two adjacent batch lanes. Keeping both predicates in one byte
// provides stable portable storage without cross-feature bit-update races.
class TwoMask {
private:
  std::uint8_t bits_ = 0;

  KOKKOS_INLINE_FUNCTION explicit TwoMask(std::uint8_t bits) : bits_(bits) {}

public:
  TwoMask() = default;

  KOKKOS_INLINE_FUNCTION static TwoMask from_bools(bool low, bool high) {
    return TwoMask(static_cast<std::uint8_t>((low ? 1u : 0u) | (high ? 2u : 0u)));
  }

  KOKKOS_INLINE_FUNCTION static TwoMask splat(bool value) { return from_bools(value, value); }

  KOKKOS_INLINE_FUNCTION bool low() const { return (bits_ & 1u) != 0; }

  KOKKOS_INLINE_FUNCTION bool high() const { return (bits_ & 2u) != 0; }

  KOKKOS_INLINE_FUNCTION static TwoMask logical_and(TwoMask left, TwoMask right) {
    return TwoMask(static_cast<std::uint8_t>(left.bits_ & right.bits_));
  }

  KOKKOS_INLINE_FUNCTION static TwoMask logical_or(TwoMask left, TwoMask right) {
    return TwoMask(static_cast<std::uint8_t>(left.bits_ | right.bits_));
  }

  KOKKOS_INLINE_FUNCTION static TwoMask logical_xor(TwoMask left, TwoMask right) {
    return TwoMask(static_cast<std::uint8_t>(left.bits_ ^ right.bits_));
  }

  KOKKOS_INLINE_FUNCTION static TwoMask logical_not(TwoMask value) {
    return TwoMask(static_cast<std::uint8_t>((~value.bits_) & 3u));
  }

  KOKKOS_INLINE_FUNCTION static TwoMask equal(TwoMask left, TwoMask right) {
    return from_bools(left.low() == right.low(), left.high() == right.high());
  }
};

static_assert(sizeof(TwoMask) == 1);

}  // namespace ponni
