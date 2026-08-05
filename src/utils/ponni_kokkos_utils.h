
#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace ponni {
  // PONNI kernels assume the rightmost index is contiguous. Keep that contract
  // in one trait so every public View-based entry point produces the same
  // compile-time diagnostic instead of accepting LayoutLeft or LayoutStride.
  template <class ViewType, bool IsView = Kokkos::is_view_v<ViewType>>
  struct is_layout_right_view : std::false_type {};

  template <class ViewType>
  struct is_layout_right_view<ViewType,true>
      : std::is_same<typename ViewType::array_layout,Kokkos::LayoutRight> {};

  template <class ViewType>
  inline constexpr bool is_layout_right_view_v = is_layout_right_view<ViewType>::value;

  template <class... ViewTypes>
  KOKKOS_INLINE_FUNCTION constexpr void require_layout_right_views() {
    static_assert((is_layout_right_view_v<ViewTypes> && ...),
                  "PONNI requires Kokkos::LayoutRight Views; copy other layouts before calling PONNI");
  }

  inline std::string my_basename(const std::string& path) { 
    size_t last_slash = path.find_last_of("/\\"); 
    if (std::string::npos == last_slash) { 
        return path; 
    }   
    return path.substr(last_slash + 1); 
  }

  template <typename T, int N> struct TypeIntToViewType { using type = typename TypeIntToViewType<T*,N-1>::type; };
  template <typename T> struct TypeIntToViewType<T,0> { using type = T; };

}

#define PONNI_AUTO_LABEL() (ponni::my_basename(__FILE__) + std::string(":") + std::to_string(__LINE__)).c_str()
#if defined(KOKKOS_ENABLE_HIP)
#define PONNI_SCOPE(a,b) auto &a = b
#elif defined(KOKKOS_ENABLE_CUDA)
#define PONNI_SCOPE(a,b) auto &a = b
#else
#define PONNI_SCOPE(a,b) auto &a = std::ref(b).get()
#endif

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  #ifndef KOKKOS_ENABLE_DEBUG
    #define KOKKOS_ENABLE_DEBUG
  #endif
#endif

namespace ponni {
  #ifdef KOKKOS_ENABLE_DEBUG
    inline constexpr bool kokkos_debug = true;
  #else
    inline constexpr bool kokkos_debug = false;
  #endif

  #ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    inline constexpr bool kokkos_bounds_debug = true;
  #else
    inline constexpr bool kokkos_bounds_debug = false;
  #endif
}

#include "ponni_TwoHalf.h"
#include "ponni_TwoMask.h"
#include "ponni_SArray.h"
#include "ponni_view_ops.h"
#include "ponni_reductions.h"
