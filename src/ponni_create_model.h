
#pragma once

namespace ponni {

  template <class... LAYERS>
  inline Inference<std::tuple<LAYERS...>> create_inference_model(LAYERS const &...layers) {
    return Inference(std::make_tuple(layers...));
  }

}


