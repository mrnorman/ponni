#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py
import keras
import numpy as np
from keras import layers


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.h5>")

    out_file = sys.argv[1]
    rng = np.random.default_rng(1234)

    inputs = keras.Input(shape=(12,), name="input")
    x = layers.Dense(10, name="dense")(inputs)
    x = layers.LeakyReLU(negative_slope=0.1, name="relu_1")(x)
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

    with h5py.File(out_file, "w") as h5:
        g0 = h5.require_group("dense/dense")
        g0.create_dataset("kernel:0", data=model.get_layer("dense").get_weights()[0])
        g0.create_dataset("bias:0", data=model.get_layer("dense").get_weights()[1])

        g1 = h5.require_group("dense_1/dense_1")
        g1.create_dataset("kernel:0", data=model.get_layer("dense_1").get_weights()[0])
        g1.create_dataset("bias:0", data=model.get_layer("dense_1").get_weights()[1])

        gt = h5.require_group("test")
        gt.create_dataset("input", data=x_test.T)
        gt.create_dataset("output", data=y_test.T)


if __name__ == "__main__":
    main()
