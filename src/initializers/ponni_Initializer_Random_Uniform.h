
#pragma once

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


    template <int N> void fill(yakl::Array<real,N,yakl::memDevice> a) const {
      auto c = a.collapse(); // Alias a's data pointer with collapsed array
      YAKL_SCOPE( lb   , this->lb   );
      YAKL_SCOPE( ub   , this->ub   );
      YAKL_SCOPE( seed , this->seed );
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , c.size() , YAKL_LAMBDA (int i) {
        yakl::Random rand(seed + i);
        c(i) = rand.genFP<real>(lb,ub);
      });
    }

  };

}


