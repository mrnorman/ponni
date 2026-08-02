
#pragma once

#include <Kokkos_Random.hpp>

namespace ponni {

  template <class real>
  struct Initializer_Random_Uniform {
    public:
    real   lb;
    real   ub;
    size_t seed;

    Initializer_Random_Uniform(real lb = -0.05 , real ub = 0.05 , size_t seed = 0 ) {
      if (seed == 0) seed = time(NULL);
      this->lb   = lb;
      this->ub   = ub;
      this->seed = seed;
    }
    ~Initializer_Random_Uniform() = default;


    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const {
      using execution_space = typename ViewType::execution_space;

      auto c = a.collapse(); // Alias a's data pointer with collapsed array
      PONNI_SCOPE( lb   , this->lb   );
      PONNI_SCOPE( ub   , this->ub   );
      PONNI_SCOPE( seed , this->seed );
      Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
      Kokkos::parallel_for( PONNI_AUTO_LABEL() , c.size() , KOKKOS_LAMBDA (int i) {
        auto rand_gen = rand_pool.get_state();
        c(i) = rand_gen.drand(lb,ub);
        rand_pool.free_state(rand_gen);
      });
    }

  };

}


