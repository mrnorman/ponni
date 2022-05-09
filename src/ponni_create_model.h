
#pragma once

namespace ponni {

  template <class... LAYERS>
  inline Model<std::tuple<LAYERS...>> create_model(LAYERS const &...layers) {
    return Model(std::make_tuple(layers...));
  }

}


