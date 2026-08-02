#pragma once

#include <Kokkos_Random.hpp>
#include <cmath>
#include <ctime>

namespace ponni {

  template <class real>
  struct Initializer_None {
    Initializer_None() = default;
    ~Initializer_None() = default;

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const { }
  };


  template <class real>
  struct Initializer_Random_Uniform {
    real   lb;
    real   ub;
    size_t seed;

    Initializer_Random_Uniform(real lb = static_cast<real>(-0.05),
                               real ub = static_cast<real>( 0.05),
                               size_t seed = 0) {
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
      Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(
          seed == 0 ? static_cast<size_t>(time(NULL)) : seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        auto rand_gen = rand_pool.get_state();
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.drand(lb, ub));
        rand_pool.free_state(rand_gen);
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
      Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed == 0 ? static_cast<size_t>(time(NULL)) : seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        auto rand_gen = rand_pool.get_state();
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.drand(lb,ub));
        rand_pool.free_state(rand_gen);
      });
    }

    template <class ViewType>
    inline void fill_normal(ViewType const & a, double mean, double stdev, size_t seed) {
      using execution_space = typename ViewType::execution_space;
      auto c = ponni::flatten(a);
      Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed == 0 ? static_cast<size_t>(time(NULL)) : seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        auto rand_gen = rand_pool.get_state();
        c(i) = static_cast<typename ViewType::non_const_value_type>(rand_gen.normal(mean, stdev));
        rand_pool.free_state(rand_gen);
      });
    }

    template <class ViewType>
    inline void fill_truncated_normal(ViewType const & a, double mean, double stdev, size_t seed, double trunc_sigma = 2.0) {
      using execution_space = typename ViewType::execution_space;
      auto c = ponni::flatten(a);
      Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed == 0 ? static_cast<size_t>(time(NULL)) : seed);
      Kokkos::parallel_for(PONNI_AUTO_LABEL(), c.size(), KOKKOS_LAMBDA (int i) {
        auto rand_gen = rand_pool.get_state();
        double v = rand_gen.normal(mean, stdev);
        while (std::abs((v - mean) / stdev) > trunc_sigma) v = rand_gen.normal(mean, stdev);
        c(i) = static_cast<typename ViewType::non_const_value_type>(v);
        rand_pool.free_state(rand_gen);
      });
    }

  }


  template <class real>
  struct Initializer_Constant {
    real value;
    Initializer_Constant(real value = static_cast<real>(0)) : value(value) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
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
      : mean(mean), stdev(stdev), seed(seed) {}

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
      : mean(mean), stdev(stdev), seed(seed) {}

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
      double fan_in = init_detail::fan_in(a);
      double stdev = std::sqrt(1.0 / fan_in);
      init_detail::fill_normal(a, 0.0, stdev, seed);
    }
  };

  template <class real>
  struct Initializer_Orthogonal {
    real gain;
    size_t seed;
    Initializer_Orthogonal(real gain = static_cast<real>(1), size_t seed = 0) : gain(gain), seed(seed) {}

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      if constexpr (ViewType::rank != 2) {
        init_detail::fill_normal(a, 0.0, 1.0, seed);
      } else {
        auto host = ponni::create_host_copy(a);
        int m = host.extent(0);
        int n = host.extent(1);
        using host_execution_space = Kokkos::DefaultHostExecutionSpace;
        Kokkos::Random_XorShift64_Pool<host_execution_space> rand_pool(
            seed == 0 ? static_cast<size_t>(time(NULL)) : seed);
        auto const host_policy = Kokkos::RangePolicy<host_execution_space>(0, m * n);
        Kokkos::parallel_for("orthogonal_fill_host", host_policy, KOKKOS_LAMBDA(int idx) {
          int i = idx / n;
          int j = idx - i * n;
          auto rand_gen = rand_pool.get_state();
          host(i,j) = static_cast<real>(rand_gen.normal(0.0, 1.0));
          rand_pool.free_state(rand_gen);
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
            for (int i = 0; i < m; i++) host(i,j) = static_cast<real>(gain * host(i,j) / norm);
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
            for (int j = 0; j < n; j++) host(i,j) = static_cast<real>(gain * host(i,j) / norm);
          }
        }

        Kokkos::deep_copy(a, host);
      }
    }
  };

}
