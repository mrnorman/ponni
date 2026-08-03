#pragma once

#include <Kokkos_Core.hpp>

namespace ponni {

// Two independent FP16 batch lanes. CUDA and HIP builds use the vendors'
// native packed representation and arithmetic; other backends retain the same
// semantics as a portable two-lane fallback.
class alignas(4) TwoHalf {
public:
  using half_type = Kokkos::Experimental::half_t;

private:
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  __half2 value_;

  KOKKOS_INLINE_FUNCTION explicit TwoHalf(__half2 value) : value_(value) {}
#else
  half_type low_;
  half_type high_;

  KOKKOS_INLINE_FUNCTION TwoHalf(half_type low, half_type high) : low_(low), high_(high) {}
#endif

public:
  TwoHalf() = default;

  KOKKOS_INLINE_FUNCTION static TwoHalf from_floats(float low, float high) {
    return from_halves(Kokkos::Experimental::cast_to_half(low),
                       Kokkos::Experimental::cast_to_half(high));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf from_halves(half_type low, half_type high) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return TwoHalf(__halves2half2(static_cast<__half>(low), static_cast<__half>(high)));
#else
    return TwoHalf(low, high);
#endif
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf splat(half_type value) { return from_halves(value, value); }

  KOKKOS_INLINE_FUNCTION static TwoHalf zero() { return from_floats(0.0f, 0.0f); }

  KOKKOS_INLINE_FUNCTION float low() const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return __half2float(__low2half(value_));
#else
    return Kokkos::Experimental::cast_from_half<float>(low_);
#endif
  }

  KOKKOS_INLINE_FUNCTION float high() const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return __half2float(__high2half(value_));
#else
    return Kokkos::Experimental::cast_from_half<float>(high_);
#endif
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf fma(TwoHalf a, TwoHalf b, TwoHalf c) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return TwoHalf(__hfma2(a.value_, b.value_, c.value_));))
    KOKKOS_IF_ON_HOST((return from_floats(a.low() * b.low() + c.low(),
                                          a.high() * b.high() + c.high());))
#else
    return from_floats(a.low() * b.low() + c.low(), a.high() * b.high() + c.high());
#endif
  }

  KOKKOS_INLINE_FUNCTION friend TwoHalf operator+(TwoHalf left, TwoHalf right) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return TwoHalf(__hadd2(left.value_, right.value_));))
    KOKKOS_IF_ON_HOST((return from_floats(left.low() + right.low(), left.high() + right.high());))
#else
    return from_floats(left.low() + right.low(), left.high() + right.high());
#endif
  }

  KOKKOS_INLINE_FUNCTION friend TwoHalf operator-(TwoHalf left, TwoHalf right) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return TwoHalf(__hsub2(left.value_, right.value_));))
    KOKKOS_IF_ON_HOST((return from_floats(left.low() - right.low(), left.high() - right.high());))
#else
    return from_floats(left.low() - right.low(), left.high() - right.high());
#endif
  }

  KOKKOS_INLINE_FUNCTION friend TwoHalf operator*(TwoHalf left, TwoHalf right) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return TwoHalf(__hmul2(left.value_, right.value_));))
    KOKKOS_IF_ON_HOST((return from_floats(left.low() * right.low(), left.high() * right.high());))
#else
    return from_floats(left.low() * right.low(), left.high() * right.high());
#endif
  }

  KOKKOS_INLINE_FUNCTION friend TwoHalf operator/(TwoHalf left, TwoHalf right) {
    return from_floats(left.low() / right.low(), left.high() / right.high());
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf relu(TwoHalf value) {
    float const low = value.low();
    float const high = value.high();
    return from_floats(low > 0.0f ? low : 0.0f, high > 0.0f ? high : 0.0f);
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf sigmoid(TwoHalf value) {
    float const low = value.low();
    float const high = value.high();
    return from_floats(1.0f / (1.0f + Kokkos::exp(-low)),
                       1.0f / (1.0f + Kokkos::exp(-high)));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf tanh(TwoHalf value) {
    return from_floats(Kokkos::tanh(value.low()), Kokkos::tanh(value.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf abs(TwoHalf value) {
    return from_floats(Kokkos::abs(value.low()), Kokkos::abs(value.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf exp(TwoHalf value) {
    return from_floats(Kokkos::exp(value.low()), Kokkos::exp(value.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf log(TwoHalf value) {
    return from_floats(Kokkos::log(value.low()), Kokkos::log(value.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf sqrt(TwoHalf value) {
    return from_floats(Kokkos::sqrt(value.low()), Kokkos::sqrt(value.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf reciprocal(TwoHalf value) {
    return from_floats(1.0f / value.low(), 1.0f / value.high());
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf pow(TwoHalf base, TwoHalf exponent) {
    return from_floats(Kokkos::pow(base.low(), exponent.low()), Kokkos::pow(base.high(), exponent.high()));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf minimum(TwoHalf left, TwoHalf right) {
    return from_floats(left.low() < right.low() ? left.low() : right.low(),
                       left.high() < right.high() ? left.high() : right.high());
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf maximum(TwoHalf left, TwoHalf right) {
    return from_floats(left.low() > right.low() ? left.low() : right.low(),
                       left.high() > right.high() ? left.high() : right.high());
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf leaky_relu(TwoHalf value, float alpha) {
    float const low = value.low();
    float const high = value.high();
    return from_floats(low >= 0.0f ? low : alpha * low, high >= 0.0f ? high : alpha * high);
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf elu(TwoHalf value, float alpha) {
    float const low = value.low();
    float const high = value.high();
    return from_floats(low >= 0.0f ? low : alpha * (Kokkos::exp(low) - 1.0f),
                       high >= 0.0f ? high : alpha * (Kokkos::exp(high) - 1.0f));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf gelu(TwoHalf value, bool approximate) {
    float const low = value.low();
    float const high = value.high();
    if (approximate) {
      float constexpr factor = 0.7978845608028654f;
      return from_floats(0.5f * low * (1.0f + Kokkos::tanh(factor * (low + 0.044715f * low * low * low))),
                         0.5f * high * (1.0f + Kokkos::tanh(factor * (high + 0.044715f * high * high * high))));
    }
    return from_floats(0.5f * low * (1.0f + Kokkos::erf(low * 0.7071067811865475f)),
                       0.5f * high * (1.0f + Kokkos::erf(high * 0.7071067811865475f)));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf silu(TwoHalf value) { return value * sigmoid(value); }

  KOKKOS_INLINE_FUNCTION static TwoHalf softplus(TwoHalf value) {
    float const low = value.low();
    float const high = value.high();
    return from_floats((low > 0.0f ? low : 0.0f) + Kokkos::log(1.0f + Kokkos::exp(-Kokkos::abs(low))),
                       (high > 0.0f ? high : 0.0f) + Kokkos::log(1.0f + Kokkos::exp(-Kokkos::abs(high))));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf hard_sigmoid(TwoHalf value, float alpha, float beta) {
    float const low = alpha * value.low() + beta;
    float const high = alpha * value.high() + beta;
    return from_floats(low < 0.0f ? 0.0f : (low > 1.0f ? 1.0f : low),
                       high < 0.0f ? 0.0f : (high > 1.0f ? 1.0f : high));
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf hard_swish(TwoHalf value) {
    return value * hard_sigmoid(value, 1.0f / 6.0f, 0.5f);
  }

  KOKKOS_INLINE_FUNCTION static TwoHalf mish(TwoHalf value) {
    return value * tanh(softplus(value));
  }
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
static_assert(sizeof(TwoHalf) == sizeof(__half2));
#endif

}  // namespace ponni
