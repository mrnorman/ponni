
#pragma once

namespace ponni {

  // Convenience Inference model creation function to allow the user to just list a bunch of successive
  // Layers.
  template <class... LAYERS>
  inline Inference<std::tuple<LAYERS...>> create_inference_model(LAYERS const &...layers) {
    return Inference(std::make_tuple(layers...));
  }

}


