#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
from keras import layers

from kokkos_nn.weights import write_ponni_file


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.ponni>")

    out_file = sys.argv[1]
    rng = np.random.default_rng(1234)

    inputs = keras.Input(shape=(12,), name="input")
    x = layers.Dense(10, name="dense")(inputs)
    x = layers.Activation("tanh", name="tanh_1")(x)
    outputs = layers.Dense(4, name="dense_1")(x)
    model = keras.Model(inputs, outputs)

    for layer_name, in_dim, out_dim in [("dense", 12, 10), ("dense_1", 10, 4)]:
        w = rng.normal(0.0, 0.1, size=(in_dim, out_dim)).astype(np.float32)
        b = rng.normal(0.0, 0.1, size=(out_dim,)).astype(np.float32)
        model.get_layer(layer_name).set_weights([w, b])

    x_test = rng.normal(0.0, 1.0, size=(1, 12)).astype(np.float32)
    y_pred = model(x_test, training=False)
    if hasattr(y_pred, "detach"):
        y_test = y_pred.detach().cpu().numpy().astype(np.float32)
    else:
        y_test = np.asarray(y_pred, dtype=np.float32)

    write_ponni_file(
        {
            "dense.kernel": model.get_layer("dense").get_weights()[0],
            "dense.bias": model.get_layer("dense").get_weights()[1],
            "dense_1.kernel": model.get_layer("dense_1").get_weights()[0],
            "dense_1.bias": model.get_layer("dense_1").get_weights()[1],
            "test.input": x_test.T,
            "test.output": y_test.T,
        },
        out_file,
        source_framework="keras",
        target="test-fixture",
    )


if __name__ == "__main__":
    main()
