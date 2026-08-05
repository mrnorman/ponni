#!/usr/bin/env python3
import sys

import numpy as np
import torch
import torch.nn as nn

from kokkos_nn.weights import write_ponni_file


class ProjectionSkipNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(2, 3, bias=True)
        self.skip = nn.Linear(2, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <output.ponni>")

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

    write_ponni_file(
        {
            # Matvec expects [num_inputs, num_outputs].
            "w_main": model.main.weight.detach().cpu().numpy().T.astype(np.float32),
            "b_main": model.main.bias.detach().cpu().numpy().astype(np.float32),
            "w_skip": model.skip.weight.detach().cpu().numpy().T.astype(np.float32),
            "b_skip": model.skip.bias.detach().cpu().numpy().astype(np.float32),
            "test.input": x_np.T,
            "test.output": y.T,
        },
        out_file,
        source_framework="pytorch",
        target="test-fixture",
    )


if __name__ == "__main__":
    main()
