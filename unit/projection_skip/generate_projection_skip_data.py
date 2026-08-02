#!/usr/bin/env python3
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn


class ProjectionSkipNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(2, 3, bias=True)
        self.skip = nn.Linear(2, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.h5>")

    out_file = sys.argv[1]

    model = ProjectionSkipNet()
    with torch.no_grad():
        model.main.weight.copy_(torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=torch.float32))
        model.main.bias.copy_(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        model.skip.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32))
        model.skip.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))

    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = model(x).detach().cpu().numpy().astype(np.float32)
    x_np = x.detach().cpu().numpy().astype(np.float32)

    with h5py.File(out_file, "w") as h5:
        # Matvec expects [num_inputs, num_outputs]
        h5.create_dataset("w_main", data=model.main.weight.detach().cpu().numpy().T.astype(np.float32))
        h5.create_dataset("b_main", data=model.main.bias.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("w_skip", data=model.skip.weight.detach().cpu().numpy().T.astype(np.float32))
        h5.create_dataset("b_skip", data=model.skip.bias.detach().cpu().numpy().astype(np.float32))

        gt = h5.require_group("test")
        gt.create_dataset("input", data=x_np.T)
        gt.create_dataset("output", data=y.T)


if __name__ == "__main__":
    main()
