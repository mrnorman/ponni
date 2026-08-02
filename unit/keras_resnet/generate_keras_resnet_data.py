#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py
import keras
import numpy as np
from keras import layers


def build_model() -> keras.Model:
    inp = keras.Input(shape=(137,), name="input")

    x = layers.Dense(20, name="dense")(inp)
    x = layers.LeakyReLU(negative_slope=0.1, name="relu_0")(x)
    saved = x

    for i in range(1, 9):
        y = layers.Dense(20, name=f"dense_{i}")(x)
        y = layers.LeakyReLU(negative_slope=0.1, name=f"relu_{i}")(y)
        x = layers.Add(name=f"add_{i}")([y, saved])
        if i <= 7:
            saved = x

    out = layers.Dense(5, name="dense_9")(x)
    return keras.Model(inp, out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.h5>")

    out_file = sys.argv[1]
    rng = np.random.default_rng(5678)

    model = build_model()

    dense_shapes = [("dense", 137, 20)] + [(f"dense_{i}", 20, 20) for i in range(1, 9)] + [("dense_9", 20, 5)]
    for layer_name, in_dim, out_dim in dense_shapes:
        w = rng.normal(0.0, 0.05, size=(in_dim, out_dim)).astype(np.float32)
        b = rng.normal(0.0, 0.05, size=(out_dim,)).astype(np.float32)
        model.get_layer(layer_name).set_weights([w, b])

    x_test = rng.normal(0.0, 1.0, size=(1, 137)).astype(np.float32)
    y_pred = model(x_test, training=False)
    if hasattr(y_pred, "detach"):
        y_test = y_pred.detach().cpu().numpy().astype(np.float32)
    else:
        y_test = np.asarray(y_pred, dtype=np.float32)

    with h5py.File(out_file, "w") as h5:
        for layer_name, _, _ in dense_shapes:
            grp = h5.require_group(f"{layer_name}/{layer_name}")
            w, b = model.get_layer(layer_name).get_weights()
            grp.create_dataset("kernel:0", data=w)
            grp.create_dataset("bias:0", data=b)

        gt = h5.require_group("test")
        gt.create_dataset("input", data=x_test.T)
        gt.create_dataset("output", data=y_test.T)


if __name__ == "__main__":
    main()
