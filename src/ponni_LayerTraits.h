#pragma once
// Included by ponni.h before the built-in layer definitions.

namespace ponni {

  // Describes whether the dynamic-View inference path may combine a layer with
  // adjacent layers. The default is deliberately conservative: an existing or
  // user-defined layer without fusion metadata remains a materialization
  // barrier and continues to use the ordinary layer-by-layer traversal.
  enum class LayerFusionKind {
    barrier,
    dense,
    pointwise
  };

  template <class Layer, class = void>
  struct LayerTraits {
    LayerFusionKind static constexpr fusion_kind = LayerFusionKind::barrier;
  };

  // A custom layer opts into fusion by declaring a public static constexpr
  // `fusion_kind`. Pointwise layers must also provide
  //
  //   static real apply_fused(real value, int feature, Params const & params);
  //
  // Dense layers must provide
  //
  //   static real compute_output(input, int feature, int batch, Params const & params);
  //
  // Both functions must be KOKKOS_INLINE_FUNCTION and device-safe. Declaring a
  // layer pointwise promises that it preserves the feature count and that each
  // output depends only on the corresponding input feature. If either promise
  // is not true, leave the layer as a barrier.
  template <class Layer>
  struct LayerTraits<Layer,std::void_t<decltype(Layer::fusion_kind)>> {
    LayerFusionKind static constexpr fusion_kind = Layer::fusion_kind;
  };

  template <class Layer>
  bool static constexpr is_dense_layer_v =
      LayerTraits<Layer>::fusion_kind == LayerFusionKind::dense;

  template <class Layer>
  bool static constexpr is_pointwise_layer_v =
      LayerTraits<Layer>::fusion_kind == LayerFusionKind::pointwise;

  // Built-in activations historically expose either apply(value) or
  // apply(value, params). Bias uses the indexed apply_fused form. Supporting
  // all three here keeps their public scalar APIs stable while presenting one
  // interface to the tuple executor.
  template <class Layer, class Real>
  KOKKOS_INLINE_FUNCTION Real apply_fused_layer(Real value, int feature,
                                                 typename Layer::Params const & params) {
    if constexpr (requires { Layer::apply_fused(value, feature, params); }) {
      return Layer::apply_fused(value, feature, params);
    } else if constexpr (requires { Layer::apply(value, params); }) {
      return Layer::apply(value, params);
    } else {
      return Layer::apply(value);
    }
  }

} // namespace ponni
