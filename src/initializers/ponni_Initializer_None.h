
#pragma once

namespace ponni {

  template <class real>
  struct Initializer_None {
    public:

    Initializer_None() = default;
    ~Initializer_None() = default;


    template <int N> void fill(yakl::Array<real,N,yakl::memDevice> a) const { }

  };

}


