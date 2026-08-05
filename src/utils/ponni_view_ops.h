
#pragma once

#include <Kokkos_Core.hpp>

namespace ponni {

  // Copy into an explicitly selected memory space. This is used by model
  // factories when rebinding layer-owned parameters to the model's memory.
  template <class MemorySpace, class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_data_type,Kokkos::LayoutRight,MemorySpace>
  create_memory_space_copy(ViewType const & view, MemorySpace const & memory_space = MemorySpace()) {
    ponni::require_layout_right_views<ViewType>();
    return Kokkos::create_mirror_view_and_copy(memory_space, view);
  }


  // Retain the established convenience name, but use Kokkos's default memory
  // space instead of a PONNI-specific allocator.
  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_data_type,
                      Kokkos::LayoutRight,
                      typename Kokkos::DefaultExecutionSpace::memory_space>
  create_device_copy(ViewType const & view) {
    ponni::require_layout_right_views<ViewType>();
    using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
    return create_memory_space_copy(view, MemorySpace());
  }


  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_data_type,Kokkos::LayoutRight,Kokkos::HostSpace>
 create_host_copy(ViewType const & view) {
    ponni::require_layout_right_views<ViewType>();
    return Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{} , view );
  }


  template <class ViewType> requires Kokkos::is_view_v<ViewType>
  inline Kokkos::View<typename ViewType::non_const_value_type *,
                      Kokkos::LayoutRight,
                      typename ViewType::memory_space,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  flatten(ViewType const & view) {
    ponni::require_layout_right_views<ViewType>();
    return Kokkos::View<typename ViewType::non_const_value_type *,
                        Kokkos::LayoutRight,
                        typename ViewType::memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(view.data(),view.size());
  }
}
