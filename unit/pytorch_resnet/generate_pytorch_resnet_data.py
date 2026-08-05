#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kokkos_nn.weights import write_ponni_file


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
        raise SystemExit(f"Usage: {sys.argv[0]} <output.ponni>")

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

    tensors = {
        "test.input": x_test_np.T,
        "test.output": y_test.T,
    }
    for index in range(1, 6):
        # Canonicalize torch.nn.Linear (output,input) into PONNI's
        # Matvec (input,output) order before serialization.
        tensors[f"fc{index}.weight"] = state[f"fc{index}.weight"].numpy().T
        tensors[f"fc{index}.bias"] = state[f"fc{index}.bias"].numpy()
    write_ponni_file(tensors, out_file, source_framework="pytorch", target="test-fixture")


if __name__ == "__main__":
    main()
