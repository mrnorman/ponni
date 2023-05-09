
#pragma once

namespace ponni {

  // Convenience Inference model creation function to allow the user to just list a bunch of successive
  // Layers.
  template <class... LAYERS>
  inline Inference<std::tuple<LAYERS...>,float> create_inference_model(LAYERS const &...layers) {
    Inference<std::tuple<LAYERS...>,float> model(std::make_tuple(layers...));
    return model;
  }

  // Convenience Inference model creation function to allow the user to just list a bunch of successive
  // Layers.
  template <class... LAYERS>
  inline Inference<std::tuple<LAYERS...>,float> create_inference_model_single_precision(LAYERS const &...layers) {
    return Inference<std::tuple<LAYERS...>,float>(std::make_tuple(layers...));
  }

  // Convenience Inference model creation function to allow the user to just list a bunch of successive
  // Layers.
  template <class... LAYERS>
  inline Inference<std::tuple<LAYERS...>,double> create_inference_model_double_precision(LAYERS const &...layers) {
    return Inference<std::tuple<LAYERS...>,double>(std::make_tuple(layers...));
  }

  // // Convenience Inference model creation function to allow the user to just list a bunch of successive
  // // Layers.
  // template <class... LAYERS>
  // inline Inference<std::tuple<LAYERS...>,half> create_inference_model_half_precision(LAYERS const &...layers) {
  //   return Inference<std::tuple<LAYERS...>,half>(std::make_tuple(layers...));
  // }

}


