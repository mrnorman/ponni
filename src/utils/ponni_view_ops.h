
#pragma once

#include <Kokkos_Core.hpp>

namespace ponni {

  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_data_type,Kokkos::LayoutRight,ponni::DeviceSpace>
  create_device_copy(ViewType const & view) {
    return Kokkos::create_mirror_view_and_copy( ponni::DeviceSpace{} , view );
  }


  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_data_type,Kokkos::LayoutRight,Kokkos::HostSpace>
 create_host_copy(ViewType const & view) {
    return Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{} , view );
  }


  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_value_type *,
                      Kokkos::LayoutRight,
                      typename ViewType::memory_space,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  flatten(ViewType const & view) {
    return Kokkos::View<typename ViewType::non_const_value_type *,
                        Kokkos::LayoutRight,
                        typename ViewType::memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(view.data(),view.size());
  }
}
