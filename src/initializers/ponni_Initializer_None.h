
#pragma once

namespace ponni {

  template <class real>
  struct Initializer_None {
    public:

    Initializer_None() = default;
    ~Initializer_None() = default;


    template <class ViewType> requires yakl::is_Array<ViewType>
    void fill(ViewType const & a) const { }

  };

}


