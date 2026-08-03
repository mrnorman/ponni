
#pragma once

#include <Kokkos_Core.hpp>

namespace ponni {
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

#include "ponni_LinearAllocator.h"
#include "ponni_DeviceSpace.h"
#include "ponni_TwoHalf.h"
#include "ponni_SArray.h"
#include "ponni_view_ops.h"
#include "ponni_reductions.h"
