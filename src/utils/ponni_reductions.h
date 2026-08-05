
#pragma once

#include <Kokkos_Core.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

#include "ponni_SArray.h"

namespace ponni {
  namespace intrinsics {

    template <class ViewType> requires Kokkos::is_view_v<ViewType>
    inline typename ViewType::non_const_value_type sum(ViewType const & view) {
      ponni::require_layout_right_views<ViewType>();
      using value_type = typename ViewType::non_const_value_type;
      auto c = ponni::flatten(view);
      value_type result;
      Kokkos::parallel_reduce( PONNI_AUTO_LABEL() ,
                               Kokkos::RangePolicy<typename ViewType::execution_space>(0,c.size()) ,
                               KOKKOS_LAMBDA (ViewType::index_type i, value_type & update) {
        update += c(i);
      } , Kokkos::Sum<value_type>(result) );
      return result;
    }


    template <class SArrayType> requires SArrayType::is_SArray
    KOKKOS_INLINE_FUNCTION typename SArrayType::non_const_value_type sum(SArrayType const & array) {
      typename SArrayType::non_const_value_type result = 0;
      for (int i = 0; i < array.size(); i++) result += array.data()[i];
      return result;
    }

  }
}
