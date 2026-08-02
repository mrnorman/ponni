
#pragma once

namespace ponni {

  template <class real>
  struct Initializer_None {
    public:

    Initializer_None() = default;
    ~Initializer_None() = default;


    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    void fill(ViewType const & a) const { }

  };

}


