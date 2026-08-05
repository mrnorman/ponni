#pragma once

namespace ponni {
namespace detail {

  // Convert a layer descriptor to the model-selected memory space. A layer in
  // another space performs its explicit semantic copy; no file or flattened
  // representation is involved in model construction.
  template <class MemorySpace, class Layer>
  inline auto rebind_layer_memory_space(Layer const & layer, MemorySpace const & memory_space) {
    using ReboundLayer = typename Layer::template rebind_memory_space<MemorySpace>;
    if constexpr (std::is_same_v<typename Layer::memory_space,MemorySpace>) {
      return ReboundLayer(layer);
    } else {
      return layer.template copy_to_memory_space<MemorySpace>(memory_space);
    }
  }


  template <class Real, class ExecutionSpace, class MemorySpace, class... Layers>
  inline auto create_model(ExecutionSpace const & execution_space,
                           MemorySpace const & memory_space,
                           Layers const &... layers) {
    static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                  "create_inference_model requires a Kokkos execution space");
    static_assert(Kokkos::is_memory_space_v<MemorySpace>,
                  "create_inference_model requires a Kokkos memory space");
    static_assert(Kokkos::SpaceAccessibility<ExecutionSpace,MemorySpace>::accessible,
                  "create_inference_model execution space cannot access its memory space");

    auto rebound_layers = std::make_tuple(rebind_layer_memory_space<MemorySpace>(layers, memory_space)...);
    using LayerTuple = decltype(rebound_layers);
    return Inference<LayerTuple,Real,ExecutionSpace,MemorySpace>(rebound_layers, execution_space);
  }

} // namespace detail


  // Default model: use Kokkos's default execution space and its native memory.
  template <class... Layers>
  inline auto create_inference_model(Layers const &... layers) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<float>(ExecutionSpace(), MemorySpace(), layers...);
  }


  // Custom execution instance with its native memory space. Storing the
  // instance preserves caller-selected streams and execution resources.
  template <class ExecutionSpace, class... Layers>
    requires Kokkos::is_execution_space_v<ExecutionSpace>
  inline auto create_inference_model(ExecutionSpace const & execution_space,
                                     Layers const &... layers) {
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<float>(execution_space, MemorySpace(), layers...);
  }


  // Custom execution instance and any memory space it can access.
  template <class ExecutionSpace, class MemorySpace, class... Layers>
    requires (Kokkos::is_execution_space_v<ExecutionSpace> && Kokkos::is_memory_space_v<MemorySpace>)
  inline auto create_inference_model(ExecutionSpace const & execution_space,
                                     MemorySpace const & memory_space,
                                     Layers const &... layers) {
    return detail::create_model<float>(execution_space, memory_space, layers...);
  }


  template <class... Layers>
  inline auto create_inference_model_single_precision(Layers const &... layers) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<float>(ExecutionSpace(), MemorySpace(), layers...);
  }


  template <class... Layers>
  inline auto create_inference_model_double_precision(Layers const &... layers) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<double>(ExecutionSpace(), MemorySpace(), layers...);
  }


  template <class... Layers>
  inline auto create_inference_model_half_precision(Layers const &... layers) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<Kokkos::Experimental::half_t>(ExecutionSpace(), MemorySpace(), layers...);
  }


  template <class... Layers>
  inline auto create_inference_model_bhalf_precision(Layers const &... layers) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    return detail::create_model<Kokkos::Experimental::bhalf_t>(ExecutionSpace(), MemorySpace(), layers...);
  }

} // namespace ponni
