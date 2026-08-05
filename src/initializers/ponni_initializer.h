#pragma once

#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <stdexcept>

namespace ponni {

  namespace init_detail {

    inline void require_finite(double value, char const * message) {
      if (!std::isfinite(value)) throw std::invalid_argument(message);
    }

    inline size_t effective_seed(size_t seed) {
      return seed == 0 ? static_cast<size_t>(time(NULL)) : seed;
    }

    // Give each flattened element its own deterministic stream. A random pool
    // can associate states with threads in a scheduling-dependent order, which
    // makes two fills with the same seed differ on a parallel backend.
    KOKKOS_INLINE_FUNCTION std::uint64_t indexed_seed(std::uint64_t seed, std::uint64_t index) {
      std::uint64_t value = seed + 0x9e3779b97f4a7c15ULL * (index + 1);
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

  }

  template <class real>
  struct Initializer_None {
    Initializer_None() = default;
    ~Initializer_None() = default;

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const { ponni::require_layout_right_views<ViewType>(); }
  };


  template <class real>
  struct Initializer_Random_Uniform {
    real   lb;
    real   ub;
    size_t seed;

    Initializer_Random_Uniform(real lb = static_cast<real>(-0.05),
                               real ub = static_cast<real>( 0.05),
                               size_t seed = 0) {
      init_detail::require_finite(static_cast<double>(lb),
                                  "Initializer_Random_Uniform lower bound must be finite");
      init_detail::require_finite(static_cast<double>(ub),
                                  "Initializer_Random_Uniform upper bound must be finite");
      if (lb > ub) throw std::invalid_argument("Initializer_Random_Uniform requires lower bound <= upper bound");
      this->lb = lb;
      this->ub = ub;
      this->seed = seed;
    }
    ~Initializer_Random_Uniform() = default;

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      using execution_space = typename ViewType::execution_space;

      auto c = ponni::flatten(a);
      real const lb = this->lb;
      real const ub = this->ub;
      if (c.size() == 0) return;
      if (lb == ub) {
        Kokkos::deep_copy(a, static_cast<typename ViewType::non_const_value_type>(lb));
        return;
      }
      std::uint64_t const fill_seed = init_detail::effective_seed(seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        Kokkos::Random_XorShift64<execution_space> rand_gen(
            init_detail::indexed_seed(fill_seed, static_cast<std::uint64_t>(i)));
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.drand(lb, ub));
      });
    }
  };


  namespace init_detail {

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION double fan_in(ViewType const & a) {
      if constexpr (ViewType::rank == 1) {
        return static_cast<double>(a.extent(0));
      } else if constexpr (ViewType::rank >= 2) {
        return static_cast<double>(a.extent(0));
      }
      return static_cast<double>(a.size());
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION double fan_out(ViewType const & a) {
      if constexpr (ViewType::rank == 1) {
        return static_cast<double>(a.extent(0));
      } else if constexpr (ViewType::rank >= 2) {
        return static_cast<double>(a.extent(1));
      }
      return static_cast<double>(a.size());
    }

    template <class ViewType>
    inline void fill_uniform(ViewType const & a, double lb, double ub, size_t seed) {
      using execution_space = typename ViewType::execution_space;
      auto c = ponni::flatten(a);
      if (c.size() == 0) return;
      if (lb == ub) {
        Kokkos::deep_copy(a, static_cast<typename ViewType::non_const_value_type>(lb));
        return;
      }
      std::uint64_t const fill_seed = effective_seed(seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        Kokkos::Random_XorShift64<execution_space> rand_gen(
            indexed_seed(fill_seed, static_cast<std::uint64_t>(i)));
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.drand(lb,ub));
      });
    }

    template <class ViewType>
    inline void fill_normal(ViewType const & a, double mean, double stdev, size_t seed) {
      using execution_space = typename ViewType::execution_space;
      auto c = ponni::flatten(a);
      if (c.size() == 0) return;
      if (stdev == 0.0) {
        Kokkos::deep_copy(a, static_cast<typename ViewType::non_const_value_type>(mean));
        return;
      }
      std::uint64_t const fill_seed = effective_seed(seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        Kokkos::Random_XorShift64<execution_space> rand_gen(
            indexed_seed(fill_seed, static_cast<std::uint64_t>(i)));
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.normal(mean, stdev));
      });
    }

    template <class ViewType>
    inline void fill_truncated_normal(ViewType const & a, double mean, double stdev, size_t seed, double trunc_sigma = 2.0) {
      using execution_space = typename ViewType::execution_space;
      auto c = ponni::flatten(a);
      if (c.size() == 0) return;
      if (stdev == 0.0) {
        Kokkos::deep_copy(a, static_cast<typename ViewType::non_const_value_type>(mean));
        return;
      }
      std::uint64_t const fill_seed = effective_seed(seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        Kokkos::Random_XorShift64<execution_space> rand_gen(
            indexed_seed(fill_seed, static_cast<std::uint64_t>(i)));
        double v = rand_gen.normal(mean, stdev);
        while (std::abs((v - mean) / stdev) > trunc_sigma) v = rand_gen.normal(mean, stdev);
        c(i) = static_cast<typename ViewType::non_const_value_type>(v);
      });
    }

  }


  template <class real>
  struct Initializer_Constant {
    real value;
    Initializer_Constant(real value = static_cast<real>(0)) : value(value) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      ponni::require_layout_right_views<ViewType>();
      Kokkos::deep_copy(a, value);
    }
  };

  template <class real> struct Initializer_Zeros : Initializer_Constant<real> {
    Initializer_Zeros() : Initializer_Constant<real>(static_cast<real>(0)) {}
  };

  template <class real> struct Initializer_Ones : Initializer_Constant<real> {
    Initializer_Ones() : Initializer_Constant<real>(static_cast<real>(1)) {}
  };

  template <class real>
  struct Initializer_Random_Normal {
    real mean;
    real stdev;
    size_t seed;
    Initializer_Random_Normal(real mean = static_cast<real>(0), real stdev = static_cast<real>(1), size_t seed = 0)
      : mean(mean), stdev(stdev), seed(seed) {
      init_detail::require_finite(static_cast<double>(mean), "Initializer_Random_Normal mean must be finite");
      init_detail::require_finite(static_cast<double>(stdev), "Initializer_Random_Normal standard deviation must be finite");
      if (stdev < static_cast<real>(0)) {
        throw std::invalid_argument("Initializer_Random_Normal standard deviation must be nonnegative");
      }
    }

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      init_detail::fill_normal(a, mean, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_Truncated_Normal {
    real mean;
    real stdev;
    size_t seed;
    Initializer_Truncated_Normal(real mean = static_cast<real>(0), real stdev = static_cast<real>(1), size_t seed = 0)
      : mean(mean), stdev(stdev), seed(seed) {
      init_detail::require_finite(static_cast<double>(mean), "Initializer_Truncated_Normal mean must be finite");
      init_detail::require_finite(static_cast<double>(stdev),
                                  "Initializer_Truncated_Normal standard deviation must be finite");
      if (stdev < static_cast<real>(0)) {
        throw std::invalid_argument("Initializer_Truncated_Normal standard deviation must be nonnegative");
      }
    }

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      init_detail::fill_truncated_normal(a, mean, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_Xavier_Uniform {
    size_t seed;
    Initializer_Xavier_Uniform(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double fan_out = init_detail::fan_out(a);
      double limit = std::sqrt(6.0 / (fan_in + fan_out));
      init_detail::fill_uniform(a, -limit, limit, seed);
    }
  };

  template <class real>
  struct Initializer_Xavier_Normal {
    size_t seed;
    Initializer_Xavier_Normal(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double fan_out = init_detail::fan_out(a);
      double stdev = std::sqrt(2.0 / (fan_in + fan_out));
      init_detail::fill_normal(a, 0.0, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_He_Uniform {
    size_t seed;
    Initializer_He_Uniform(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double limit = std::sqrt(6.0 / fan_in);
      init_detail::fill_uniform(a, -limit, limit, seed);
    }
  };

  template <class real>
  struct Initializer_He_Normal {
    size_t seed;
    Initializer_He_Normal(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double stdev = std::sqrt(2.0 / fan_in);
      init_detail::fill_normal(a, 0.0, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_Lecun_Uniform {
    size_t seed;
    Initializer_Lecun_Uniform(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double limit = std::sqrt(3.0 / fan_in);
      init_detail::fill_uniform(a, -limit, limit, seed);
    }
  };

  template <class real>
  struct Initializer_Lecun_Normal {
    size_t seed;
    Initializer_Lecun_Normal(size_t seed = 0) : seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if (a.size() == 0) return;
      double fan_in = init_detail::fan_in(a);
      double stdev = std::sqrt(1.0 / fan_in);
      init_detail::fill_normal(a, 0.0, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_Orthogonal {
    real gain;
    size_t seed;
    Initializer_Orthogonal(real gain = static_cast<real>(1), size_t seed = 0) : gain(gain), seed(seed) {
      init_detail::require_finite(static_cast<double>(gain), "Initializer_Orthogonal gain must be finite");
    }

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      ponni::require_layout_right_views<ViewType>();
      if (a.size() == 0) return;
      if constexpr (ViewType::rank != 2) {
        init_detail::fill_normal(a, 0.0, 1.0, seed);
      } else {
        auto host = ponni::create_host_copy(a);
        int m = host.extent(0);
        int n = host.extent(1);
        using host_execution_space = Kokkos::DefaultHostExecutionSpace;
        std::uint64_t const fill_seed = init_detail::effective_seed(seed);
        auto const host_policy = Kokkos::RangePolicy<host_execution_space>(0, m * n);
        Kokkos::parallel_for("orthogonal_fill_host", host_policy, KOKKOS_LAMBDA(int idx) {
          int i = idx / n;
          int j = idx - i * n;
          Kokkos::Random_XorShift64<host_execution_space> rand_gen(
              init_detail::indexed_seed(fill_seed, static_cast<std::uint64_t>(idx)));
          host(i,j) = static_cast<real>(rand_gen.normal(0.0, 1.0));
        });
        Kokkos::fence("orthogonal_fill_host complete");

        if (m >= n) {
          for (int j = 0; j < n; j++) {
            for (int k = 0; k < j; k++) {
              double dot = 0;
              for (int i = 0; i < m; i++) dot += host(i,j) * host(i,k);
              for (int i = 0; i < m; i++) host(i,j) -= static_cast<real>(dot * host(i,k));
            }
            double norm = 0;
            for (int i = 0; i < m; i++) norm += host(i,j) * host(i,j);
            norm = std::sqrt(norm) + 1.e-20;
            for (int i = 0; i < m; i++) host(i,j) = static_cast<real>(host(i,j) / norm);
          }
        } else {
          for (int i = 0; i < m; i++) {
            for (int k = 0; k < i; k++) {
              double dot = 0;
              for (int j = 0; j < n; j++) dot += host(i,j) * host(k,j);
              for (int j = 0; j < n; j++) host(i,j) -= static_cast<real>(dot * host(k,j));
            }
            double norm = 0;
            for (int j = 0; j < n; j++) norm += host(i,j) * host(i,j);
            norm = std::sqrt(norm) + 1.e-20;
            for (int j = 0; j < n; j++) host(i,j) = static_cast<real>(host(i,j) / norm);
          }
        }

        // Gram-Schmidt projections assume previously accepted vectors have
        // unit norm. Applying gain during normalization corrupts subsequent
        // projections whenever gain differs from one, so scale only after the
        // complete orthonormal basis has been constructed.
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) host(i,j) *= gain;
        }

        Kokkos::deep_copy(a, host);
      }
    }
  };

}
