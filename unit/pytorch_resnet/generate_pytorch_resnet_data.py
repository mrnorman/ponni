#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(12, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 5)
        self.fc5 = nn.Linear(5, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        saved = x

        x = F.silu(self.fc2(x))
        x = x + saved
        saved = x

        x = F.silu(self.fc3(x))
        x = x + saved
        saved = x

        x = F.silu(self.fc4(x))
        x = x + saved

        return self.fc5(x)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.h5>")

    out_file = sys.argv[1]

    torch.manual_seed(9012)
    np_rng = np.random.default_rng(9012)

    model = TinyResNet()
    for param in model.parameters():
        nn.init.uniform_(param, a=-0.2, b=0.2)

    x_test_np = np_rng.normal(0.0, 1.0, size=(1, 12)).astype(np.float32)
    x_test = torch.from_numpy(x_test_np)
    y_test = model(x_test).detach().numpy().astype(np.float32)

    state = model.state_dict()

    with h5py.File(out_file, "w") as h5:
        h5.create_dataset("0.0.0.0.1.weight", data=state["fc1.weight"].numpy())
        h5.create_dataset("0.0.0.0.1.bias", data=state["fc1.bias"].numpy())

        h5.create_dataset("0.0.0.2.sequential.0.weight", data=state["fc2.weight"].numpy())
        h5.create_dataset("0.0.0.2.sequential.0.bias", data=state["fc2.bias"].numpy())

        h5.create_dataset("0.0.2.sequential.0.weight", data=state["fc3.weight"].numpy())
        h5.create_dataset("0.0.2.sequential.0.bias", data=state["fc3.bias"].numpy())

        h5.create_dataset("0.2.sequential.0.weight", data=state["fc4.weight"].numpy())
        h5.create_dataset("0.2.sequential.0.bias", data=state["fc4.bias"].numpy())

        h5.create_dataset("2.weight", data=state["fc5.weight"].numpy())
        h5.create_dataset("2.bias", data=state["fc5.bias"].numpy())

        gt = h5.require_group("test")
        gt.create_dataset("input", data=x_test_np.T)
        gt.create_dataset("output", data=y_test.T)


if __name__ == "__main__":
    main()
