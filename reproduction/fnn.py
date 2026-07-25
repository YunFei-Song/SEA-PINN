"""FNN-PINN baseline network."""

from __future__ import annotations

from typing import List

import deepxde as dde
import torch


def create_fnn(layer_sizes: List[int]):
    """Create the FNN baseline used in the reproduction experiments."""

    net = dde.nn.FNN(layer_sizes, "silu", "zeros")
    for linear in net.linears:
        torch.nn.init.uniform_(linear.weight, a=-0.5, b=0.5)
        torch.nn.init.zeros_(linear.bias)
    return net
